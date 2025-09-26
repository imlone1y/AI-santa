import os
import time
import asyncio
import tempfile
import wave
from dotenv import load_dotenv
import tiktoken
import json
from datetime import datetime
import numpy as np
import sounddevice as sd
import requests
from deepgram import Deepgram
from openwakeword.model import Model
from openwakeword import utils
import pyaudio
import platform

# 初始化喚醒詞模型
utils.download_models()
wakeword_model = Model(
    wakeword_models=["./hey_santa.tflite"],  # 你的模型路徑
    inference_framework="tflite"
)

# --------------------------------------------------
# Load API keys
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DG = Deepgram(DEEPGRAM_API_KEY)


# --------------------------------------------------
# Audio constants
# --------------------------------------------------
RATE = 16000            # 16 kHz mono
FRAME_DURATION = 0.2     # seconds per audio chunk
MAX_RECORD_SEC = 30      # hard stop after 30 s
MIN_RECORD_SEC = 1
SILENCE_SEC = 0.7     # stop after this long of silence
SILENCE_THRESHOLD = 500   # RMS amplitude threshold (0-32767)

# count token
def count_gpt_tokens(messages, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = 0
    for m in messages:
        tokens += 4  # 每則訊息基本格式 token
        tokens += len(encoding.encode(m.get("content", "")))
    tokens += 2  # 結尾 assistant prefix
    return tokens

def save_usage_log(prompt, reply, gpt_tokens, tts_chars):
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "reply": reply,
        "gpt_tokens": gpt_tokens,
        "tts_characters": tts_chars
    }
    log_file = "usage_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

# --------------------------------------------------
# wake word
# --------------------------------------------------
def _flush_stream():
    """清空 PyAudio 輸入 buffer。"""
    for _ in range(5):
        try:
            stream.read(1600, exception_on_overflow=False)
        except Exception:
            pass


def listen_for_wakeword():
    CHUNK = 1600
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    THRESHOLD = 0.7
    REQUIRED_HITS = 1  # 必須連續命中幾次
    consecutive_hits = 0

    global stream
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("🎧 Listening for wakeword 'hey_santa'...")

    try:
        while True:
            pcm = stream.read(CHUNK, exception_on_overflow=False)
            scores = wakeword_model.predict(np.frombuffer(pcm, dtype=np.int16))
            score = scores.get("hey_santa", 0.0)  # 根據你模型名稱改這行

            if score > THRESHOLD:
                consecutive_hits += 1
                if consecutive_hits >= REQUIRED_HITS:
                    print(f"✅ Wake word detected! Score={score:.2f}")
                    wakeword_model.reset()
                    _flush_stream()
                    hoho_path = "hello.mp3"
                    if os.path.exists(hoho_path):
                        if platform.system() == "Darwin":
                            os.system(f"afplay {hoho_path}")
                        elif platform.system() == "Windows":
                            os.system(f"start {hoho_path}")
                        else:
                            os.system(f"mpg123 {hoho_path}")  # Linux
                    else:
                        print("⚠️ hello.mp3 not found!")
                    return
            else:
                consecutive_hits = 0
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()




# --------------------------------------------------
# Helper – calculate RMS volume for silence detection
# --------------------------------------------------

def rms(int16_audio: np.ndarray) -> float:
    if int16_audio.size == 0:
        return 0.0
    return np.sqrt(np.mean(int16_audio.astype(np.float32) ** 2))

# --------------------------------------------------
# Record until silence
# --------------------------------------------------

def record_until_silence() -> str:
    """Record microphone input until silence detected. Returns WAV filename."""
    blocksize = int(RATE * FRAME_DURATION)
    silent_blocks = 0
    required_silent = int(SILENCE_SEC / FRAME_DURATION)
    frames = []

    print("🎙️ Recording… Speak now!")
    start = time.time()
    with sd.InputStream(samplerate=RATE, channels=1, dtype='int16', blocksize=blocksize) as stream:
        while True:
            chunk, _ = stream.read(blocksize)
            frames.append(chunk.copy())
            level = rms(chunk)

            # defer silence detection until min speech time passed
            if time.time() - start >= MIN_RECORD_SEC:
                if level < SILENCE_THRESHOLD:
                    silent_blocks += 1
                else:
                    silent_blocks = 0

            # stop if long enough silence or max length reached
            if silent_blocks >= required_silent or time.time() - start >= MAX_RECORD_SEC:
                break

    audio = np.concatenate(frames, axis=0)
    outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(outfile.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio.tobytes())

    duration = len(audio) / RATE
    # print(f"✅ 錄音完成：{outfile.name}  (≈{duration:.1f}s)")
    return outfile.name

# --------------------------------------------------
# Whisper transcription (auto language)
# --------------------------------------------------


async def transcribe_with_deepgram(wav_path: str) -> str:
    # print("🔍 使用 Deepgram v2 辨識中…")
    with open(wav_path, "rb") as f:
        source = {
            "buffer": f.read(),
            "mimetype": "audio/wav"
        }
    options = {
        "model": "nova-3",
        "language": "en-US",
        "smart_format": False
    }
    
    resp = await DG.transcription.prerecorded(source, options)
    transcript = resp["results"]["channels"][0]["alternatives"][0]["transcript"]
    print("📝 Deepgram 結果：", transcript)
    return transcript.strip()

# --------------------------------------------------
# ChatGPT response with memory
# --------------------------------------------------

def chat_response(prompt: str, memory: list[str]) -> str:
    memory.append({"role": "user", "content": prompt})
    print("💬 GPT 回應中…")
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": memory,
            "max_tokens": 150,
        }
    )
    resp.raise_for_status()
    reply = resp.json()["choices"][0]["message"]["content"].strip()
    print("🤖 GPT：", reply)
    memory.append({"role": "assistant", "content": reply})
    return reply

# --------------------------------------------------
# ElevenLabs TTS
# --------------------------------------------------

def speak(reply: str):
    voice_id = os.getenv("VOICE_ID")  # 可換成自訂聖誕老人聲
    print("🗣️ TTS 合成中…")
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={
            "text": reply,
            "model_id": "eleven_flash_v2_5",
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.55},
        },
    )
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(resp.content)
        # macOS 用 afplay, Windows 用 start, Linux 用 mpv/aplay 等
        os.system(f"afplay {f.name}" if os.name == 'posix' else f"start {f.name}")

# --------------------------------------------------
# Main event loop
# --------------------------------------------------
async def main():
    memory = [{"role": "system", "content": "你是溫暖幽默的聖誕老人，回答要親切簡短。用英文回覆。"}]

    while True:
        # listen_for_wakeword()  # 等待喚醒詞
        # print("🎙️ 開始對話囉～")

        while True:
            wav_file = record_until_silence()
            user_text = await transcribe_with_deepgram(wav_file)

            if not user_text:
                print("❌ 沒有辨識到語音，請再說一次。")
                continue

            if any(kw in user_text.lower() for kw in ("bye", "goodbye", "離開", "退出", "再見")):
                # print("👋 Ho-ho-ho，再見！等待下一次喚醒...")
                hoho_path = "bye.mp3"
                if os.path.exists(hoho_path):
                    if platform.system() == "Darwin":
                        os.system(f"afplay {hoho_path}")
                    elif platform.system() == "Windows":
                        os.system(f"start {hoho_path}")                        
                    else:
                        os.system(f"mpg123 {hoho_path}")  # Linux
                else:
                    print("⚠️ bye.mp3 not found!")
                break  # 回到喚醒狀態

            reply = chat_response(user_text, memory)
            speak(reply)

            # gpt_token_count = count_gpt_tokens(memory)
            # tts_char_count = len(reply)
            # save_usage_log(user_text, reply, gpt_token_count, tts_char_count)
            # print(f"🧮 Token 使用：GPT={gpt_token_count} / ElevenLabs 字數={tts_char_count}")



if __name__ == "__main__":
    asyncio.run(main())
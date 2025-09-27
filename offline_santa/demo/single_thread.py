from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import speech_recognition as sr
import tempfile
from faster_whisper import WhisperModel
import sys
from TTS.api import TTS
import tempfile
import os
import time
from pydub import AudioSegment
from pydub.playback import play
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


xtts_model = TTS(
    model_path="./trainedmodel/",
    config_path="./trainedmodel/config.json",
    gpu=False
)


def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

local_model_path = get_resource_path("santa-merged")
END_CONVERSATION_COMMAND = "end conversation"


def initialize_components():
    print(f"Loading LLM from '{local_model_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(local_model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        print("LLM loaded.")
    except Exception as e:
        print(f"LLM load error: {e}")
        return None, None, None, None, None, None

    print("Initializing microphone and STT model...")
    try:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Microphone and Whisper ready.")
    except Exception as e:
        print(f"Initialization error: {e}")
        return None, None, None, None, None, None

    return tokenizer, model, pipe, device, recognizer, microphone, whisper_model



def listen_and_transcribe(recognizer, microphone, whisper_model):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("\n🎤 Speak now:")
        audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_wav.write(audio_data.get_wav_data())
        temp_path = temp_wav.name

    try:
        segments, _ = whisper_model.transcribe(temp_path)
        user_text = " ".join([s.text for s in segments]).strip()
    except Exception as e:
        print(f"[Whisper STT] Error: {e}")
        user_text = None
    finally:
        os.remove(temp_path)

    return user_text



def speak_with_xtts(text):
    try:
        output_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        xtts_model.tts_to_file(
            text=text,
            speaker_wav="./trainedmodel/wavs/santa001_00000017.wav",
            language="en",
            file_path=output_wav
        )

        # ✅ 載入並播放（會等待播放結束）
        audio = AudioSegment.from_wav(output_wav)
        play(audio)
        os.remove(output_wav)

    except Exception as e:
        print(f"[XTTS TTS] Error: {e}")



if __name__ == "__main__":
    tokenizer, model, pipe, device, recognizer, microphone, whisper_model = initialize_components()
    if not all([tokenizer, model, pipe, device, recognizer, microphone, whisper_model]):
        print("Initialization failed.")
        exit()

    print(f"\n🎄 Ready! Say '{END_CONVERSATION_COMMAND}' to exit.\n")

    while True:
        try:
            user_input = listen_and_transcribe(recognizer, microphone, whisper_model)
            if not user_input:
                continue

            print(f"[User] {user_input}")
            if END_CONVERSATION_COMMAND in user_input.lower():
                print("🎅 Exiting...")
                break

            messages = [
                {"role": "system", "content": "You are Santa Claus."},
                {"role": "user", "content": user_input},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
            response = output[0]["generated_text"].split("<|assistant|>")[-1].strip()

            print(f"[Santa 🎅] {response}")

            speak_with_xtts(response)

        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt]")
            break
        except Exception as e:
            print(f"[Main Error] {e}")
            continue

    print("👋 Bye!")

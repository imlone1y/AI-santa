from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import speech_recognition as sr
import time
import tempfile
from faster_whisper import WhisperModel
import threading
import queue
import sys
import subprocess
from TTS.api import TTS
import tempfile
import os

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

stt_queue = queue.Queue()
mic_ready_event = threading.Event()
tts_interrupt_event = threading.Event()
stop_threads_event = threading.Event()
active_tts_thread = None

llm_tokenizer = None
llm_model = None
llm_device = None
stt_recognizer = None
stt_microphone = None
whisper_stt_model = None

local_model_path = get_resource_path("santa-tinyllama-full-merged")
END_CONVERSATION_COMMAND = "end conversation"


def initialize_components():
    global llm_tokenizer, llm_model, llm_device, stt_recognizer, stt_microphone, whisper_stt_model

    print(f"Loading LLM from '{local_model_path}'...")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_model = AutoModelForCausalLM.from_pretrained(local_model_path)
        llm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llm_model.to(llm_device)
        print("LLM loaded.")
    except Exception as e:
        print(f"LLM load error: {e}")
        return False

    print("Initializing microphone...")
    try:
        stt_recognizer = sr.Recognizer()
        stt_microphone = sr.Microphone()
        print("Microphone ready.")
    except Exception as e:
        print(f"Microphone error: {e}")
        return False

    print("Loading faster-whisper model...")
    try:
        whisper_stt_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Whisper model ready.")
    except Exception as e:
        print(f"Whisper load error: {e}")
        return False

    return True


def microphone_listener_thread_func():
    global active_tts_thread

    while not stop_threads_event.is_set():
        is_speaking = active_tts_thread is not None and active_tts_thread.is_alive()
        listen_mode = "Interrupt Detection" if is_speaking else "Normal Listening"
        audio_data = None

        with stt_microphone as source:
            try:
                if listen_mode == "Normal Listening":
                    if not mic_ready_event.is_set():
                        mic_ready_event.wait(timeout=0.5)
                    mic_ready_event.clear()
                    stt_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    print("\n🎤 Speak now:")
                    listen_timeout = 5
                    listen_phrase_limit = 15
                else:
                    listen_timeout = 0.5
                    listen_phrase_limit = 3

                audio_data = stt_recognizer.listen(source, timeout=listen_timeout, phrase_time_limit=listen_phrase_limit)
            except sr.WaitTimeoutError:
                if listen_mode == "Normal Listening":
                    mic_ready_event.set()
                continue
            except Exception as e:
                print(f"[Microphone] Error: {e}")
                if listen_mode == "Normal Listening":
                    mic_ready_event.set()
                time.sleep(0.5)
                continue

        if audio_data:
            user_text = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                temp_wav.write(audio_data.get_wav_data())
                temp_path = temp_wav.name

            try:
                segments, _ = whisper_stt_model.transcribe(temp_path)
                user_text = " ".join([s.text for s in segments]).strip()
            except Exception as e:
                print(f"[Whisper STT] Error: {e}")
            finally:
                os.remove(temp_path)

            if user_text:
                print(f"[User] {user_text}")
                if is_speaking:
                    tts_interrupt_event.set()
                stt_queue.put(user_text)
            else:
                mic_ready_event.set()



def tts_playback_thread_func(text_to_speak):
    global mic_ready_event
    try:
        output_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        xtts_model.tts_to_file(
            text=text_to_speak,
            speaker_wav="./trainedmodel/wavs/santa001.wav",  # 你的 Santa 語音樣本
            language="en",
            file_path=output_wav
        )
        os.system(f"aplay {output_wav}")
        os.remove(output_wav)
    except Exception as e:
        print(f"[XTTS TTS] Error: {e}")
    finally:
        if not stop_threads_event.is_set():
            mic_ready_event.set()


if __name__ == "__main__":
    if not initialize_components():
        print("Initialization failed.")
        exit()

    print(f"\n🎄 Ready! Say '{END_CONVERSATION_COMMAND}' to exit.\n")

    mic_thread = threading.Thread(target=microphone_listener_thread_func, daemon=True)
    mic_thread.start()
    mic_ready_event.set()

    try:
        while not stop_threads_event.is_set():
            try:
                user_input_text = stt_queue.get(timeout=1)
                if user_input_text is None and stop_threads_event.is_set():
                    break
                if END_CONVERSATION_COMMAND in user_input_text.lower():
                    print("🎅 Exiting...")
                    stop_threads_event.set()
                    mic_ready_event.set()
                    if active_tts_thread and active_tts_thread.is_alive():
                        active_tts_thread.join(timeout=2)
                    break

                if active_tts_thread and active_tts_thread.is_alive():
                    active_tts_thread.join(timeout=2)

                print("[LLM] Generating reply...")
                prompt = f"You are Santa Claus. Speak like a jolly, wise man.\nUser: {user_input_text}\nSanta:"
                inputs = llm_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                input_ids = inputs['input_ids'].to(llm_device)
                attention_mask = inputs['attention_mask'].to(llm_device)

                with torch.no_grad():
                    output = llm_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=40,
                        pad_token_id=llm_tokenizer.eos_token_id,
                        eos_token_id=llm_tokenizer.eos_token_id,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        temperature=0.5,
                        top_k=20
                    )

                response = llm_tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True).strip()
                print(f"[Santa 🎅] {response}")
                if response:
                    tts_interrupt_event.clear()
                    mic_ready_event.clear()
                    active_tts_thread = threading.Thread(target=tts_playback_thread_func, args=(response,), daemon=True)
                    active_tts_thread.start()
                else:
                    mic_ready_event.set()
                stt_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Main Error] {e}")
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt]")
    finally:
        print("[Cleanup]")
        stop_threads_event.set()
        mic_ready_event.set()
        try:
            stt_queue.put(None, timeout=0.1)
        except:
            pass
        if active_tts_thread and active_tts_thread.is_alive():
            active_tts_thread.join(timeout=3)
        if mic_thread and mic_thread.is_alive():
            mic_thread.join(timeout=3)
        print("👋 Bye!")

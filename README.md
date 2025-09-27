# AI Santa Claus

[繁體中文](README_TW.md) | English

This project is currently under development. Unauthorized use or sale is strictly prohibited.

## Project Introduction

The goal of this project is to develop a conversational AI Santa Claus and deploy it on an Orange Pi (orangepi), making it an independent AI toy. The project includes two versions: **online** and **offline**.

## Online Version

The `santa_demo` folder contains the online version, which cannot run offline. To run tests, you must provide the required service API KEYS in the `.env` file. This project has been tested on `macOS 15.6.1` with `python 3.13.3`.

### Project Structure

```
.
├── .env              # Environment variables
├── bye.mp3           # Voice output when ending the conversation
├── hello.mp3         # Voice output when waking up
├── hey_santa.onnx    # Wake word model trained with OpenWakeWord
├── hey_santa.tflite  # Wake word model trained with OpenWakeWord
├── main.py           # Full dialogue pipeline using multiple online APIs
├── requirements.txt  
└── santa.py          # Using the OpenAI Realtime API service
```

### `main.py` Workflow

<img width="1827" height="1347" alt="online workflow" src="https://github.com/user-attachments/assets/c245f999-543a-4ce3-ad8d-57873f2868ab" />

---

## Offline Version

The `offline_santa` folder contains the offline version, which includes model fine-tuning, speech synthesis, and the complete dialogue pipeline. Model fine-tuning and TTS training were performed on an `RTX 3060 Laptop` and `Intel i9 12th`, tested with `python 3.10.18`.

### Project Structure

```
.
├── demo                            # Conversation
│   ├── multi_threads.py            # Multi-threaded version; supports simultaneous speech output and listening, with interruption
│   ├── requirements.txt
│   ├── santa_chat.py               # Text input for testing model responses
│   ├── santa_demo.wav              # Example output audio from speech model
│   ├── santa-merged                # Language model storage
│   │   └── temp.txt
│   ├── single_thread.py            # Single-threaded conversation version, more stable
│   └── trainedmodel                # Speech model storage
│       └── temp.txt
└── santa_train                     # Training
    ├── combine.py                  # Merge fine-tuned model with base model
    ├── santa_chat.py               # Text input for testing model responses
    ├── santa-lora-trl-3000         # Fine-tuned model output directory
    │   └── temp.txt
    ├── santa-merged                # Merged model output directory
    │   └── temp.txt
    ├── train                       # Fine-tuning dialogue datasets
    │   ├── santa_chat_3000.json
    │   ├── santa_chat2.json
    │   └── santa_chat3.json
    └── train_llama.py              # Fine-tuning script
```

### Training Instructions

* The base language model used is `TinyLlama-1.1B-Chat-v1.0`, fine-tuned with SFT and LoRA.
  Before fine-tuning, modify `line 11` in `./santa_train/train_llama.py` with your own `HUGGING_FACE_TOKEN`.

* The speech model was trained using [coqui TTS](https://github.com/coqui-ai/TTS), with the [alltalk tts](https://github.com/erew123/alltalk_tts) framework.

After training, run `./santa_train/combine.py` to merge the fine-tuned model with the base model. Only the merged model can be used by other components. The merged model will be stored under `./santa_train/santa_merged/`.
Replace the `./demo/santa_merged/` folder with this merged directory to enable full conversation functionality.

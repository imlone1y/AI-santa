# AI 聖誕老人

繁體中文 | [English](README.md)

本項目處於開發階段，未經授權禁止使用、販售。

## 項目介紹

本項目為開發一款可對話之 AI 聖誕老人，並將程式燒錄於香橙派 ( orangepi ) 上，成為可獨立運行的 AI 玩具。項目包含**需連網**以及**離線**兩個版本。

## 連網方法

`santa_demo` 資料夾底下為連網版本，無法離線執行，執行測試需需提供 `.env` 檔案裡所需服務之 API KEYS。本項目在 `macOS 15.6.1` 上，以 `python 3.13.3` 版本完成測試。

### 項目結構
```
.
├── .env              # 環境變數
├── bye.mp3           # 結束對話時語音 
├── hello.mp3         # 喚醒時語音
├── hey_santa.onnx    # 使用 OpenWakeWord 訓練出喚醒詞檔案
├── hey_santa.tflite  # 使用 OpenWakeWord 訓練出喚醒詞檔案
├── main.py           # 使用多個線上 API 串接完整對話流程
├── requirements.txt  
└── santa.py          # 使用 OpenAI Realtime API 服務
```

### `main.py` 流程

<img width="1827" height="1347" alt="online workflow" src="https://github.com/user-attachments/assets/1ac76819-be06-48a4-863a-ffc5e2b8b2c9" />

---

## 離線方法

`offline_santa` 資料夾底下為離線版本，其底下包涵模型微調、語音合成及完整對話流程程式碼。本項目微調、合成語音訓練均使用 `RTX 3060 Laptop` 及 `Intel i9 12th` 訓練，以 `python 3.10.18` 版本完成測試。

### 項目結構
```
.
├── demo                            # 對話
│   ├── multi_threads.py            # 多線程版本，可邊輸出語音邊監聽，並及時打斷
│   ├── requirements.txt
│   ├── santa_chat.py               # 文字輸入，測試模型回覆狀況
│   ├── santa_demo.wav              # 語音模型輸出音檔
│   ├── santa-merged                # 語言模型放置處
│   │   └── temp.txt
│   ├── single_thread.py            # 單線程對話版本，較穩定
│   └── trainedmodel                # 語音模型放置處
│       └── temp.txt
└── santa_train                     # 訓練
    ├── combine.py                  # 將微調完成模型與基底模型合併
    ├── santa_chat.py               # 文字輸入，測試模型回覆狀況
    ├── santa-lora-trl-3000         # 微調完成模型輸出處
    │   └── temp.txt
    ├── santa-merged                # 合併完成模型輸出處
    │   └── temp.txt
    ├── train                       # 微調用對話訓練集
    │   ├── santa_chat_3000.json
    │   ├── santa_chat2.json
    │   └── santa_chat3.json
    └── train_llama.py              # 微調程式碼
```

### 訓練說明

- 語言模型基底使用 `TinyLlama-1.1B-Chat-v1.0` 進行微調，方法使用SFT 與 LoRA 進行。微調前，需將 `./santa_train/train_llama.py` `第 11 行`需改為自己的 `HUGGING_FACE_TOKEN`。
- 語音模型訓練使用 [coqui TTS](https://github.com/coqui-ai/TTS)，框架使用 [alltalk tts](https://github.com/erew123/alltalk_tts) 進行模型訓練。

訓練完成後，需執行 `./santa_train/combine.py`，將模型與基底模型合併，才能輸出完整模型供其他元件調用。合併後模型存於 `./santa_train/santa_merged/` 底下，將資料夾取代 `./demo/santa_merged/` 後，方可進行對話。

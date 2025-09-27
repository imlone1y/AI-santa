from TTS.api import TTS

# 載入你訓練好的模型（記得加上 config 路徑）
tts = TTS(
    model_path="./trainedmodel/",
    config_path="./trainedmodel/config.json",
    gpu=False  # 如果你要用 CUDA，否則改成 False
)

# 產生語音
tts.tts_to_file(
    text="i like child, merry chrismas!",
    file_path="santa_demo.wav",
    speaker_wav="./trainedmodel/wavs/santa001.wav",
    language="en"
)

import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 模型與 Token 設定
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
hf_token = "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"  # ← 替換為你的 token

# 載入模型與 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=hf_token)

# LoRA 設定
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 載入 train 資料夾下所有 json 資料
train_folder = "./train"
all_data = []
for file_name in os.listdir(train_folder):
    if file_name.endswith(".json"):
        file_path = os.path.join(train_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)

dataset = Dataset.from_list(all_data)

# Chat 模板轉換
def formatting_prompts(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}

dataset = dataset.map(formatting_prompts)

# LoRA + TRL 微調設定
sft_config = SFTConfig(
    output_dir="./santa-lora-trl-3000",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    logging_steps=20,
    save_strategy="epoch",
    learning_rate=5e-5,
    fp16=False,
    packing=False,
    max_seq_length=256
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer  # ✅ 改用這個參數
)


# 開始訓練
trainer.train()

# 儲存模型
trainer.model.save_pretrained("./santa-lora-trl-3000")
tokenizer.save_pretrained("./santa-lora-trl-3000")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# base model 和 LoRA adapter 路徑
base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_path = "./santa-lora-trl-3000"
output_path = "./santa-merged"

# 載入 base model 與 LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_path)

# 合併 LoRA 權重進 base model
model = model.merge_and_unload()

# 載入對應 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 儲存合併後模型與 tokenizer
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("✅ 合併完成，儲存於:", output_path)

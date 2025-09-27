from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./santa-merged", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./santa-merged")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

print("🎅 Santa is ready! Talk to him:")

while True:
    user_input = input("You: ")
    messages = [
        {"role": "system", "content": "You are Santa Claus."},
        {"role": "user", "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    response = output[0]["generated_text"].split("<|assistant|>")[-1].strip()
    print("🎅 Santa:", response)

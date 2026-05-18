from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(UserSecretsClient().get_secret("HF_TOKEN"))


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id="meta-llama/Llama-3.1-8B-Instruct"
adapther_path="/kaggle/input/datasets/akiremkoffi/model-brain"

bnb_config= BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

print('load the brain')
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Attaching Fine-Tuned Adapters...")
model = PeftModel.from_pretrained(base_model, adapther_path)
model.eval()

test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the AI Agentic Career Architect, a specialized assistant for academic and career research.<|eot_id|><|start_header_id|>user<|end_header_id|>
Explain how your background in high-pressure service environments contributes to your grit as a Machine Learning Engineer.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

print("\n--- GENERATING RESPONSE ---\n")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        temperature=0.7, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("assistant")[-1].strip())
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
import torch
import gc
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# --- login ---
login(UserSecretsClient().get_secret("HF_TOKEN"))

# --- tunables ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "/kaggle/input/datasets/akiremkoffi/dataset/training_data.jsonl"
OUT_DIR = "/kaggle/working/results"
MERGED_DIR = "/kaggle/working/merged_career_architect"

def format_ai_prompt(batch):
    """One function, module-level: avoids class/self confusion; handles batched=True."""
    texts = []
    for m in batch["messages"]:
        texts.append(
            f"<|system|>\n{m[0]['content']}\n<|end|>\n"
            f"<|user|>\n{m[1]['content']}\n<|end|>\n"
            f"<|assistant|>\n{m[2]['content']}\n<|end|>\n"
        )
    return {"text": texts}

class Brain:
    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        self.bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def load(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.bnb,
            device_map="auto",
            token=True
        )
        tok = AutoTokenizer.from_pretrained(self.model_id, token=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return model, tok

class Trainer:
    def run(self, model, tokenizer, data_path=DATA_PATH, out_dir=OUT_DIR):
        lora = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        ds = load_dataset("json", data_files=data_path, split="train").map(format_ai_prompt, batched=True)
        split = ds.train_test_split(test_size=0.1, seed=42)

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora)

        args = SFTConfig(
            output_dir=out_dir,
            eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100, max_length=512,
            num_train_epochs=3, learning_rate=2e-4, dataset_text_field="text",
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            report_to="none" # Prevents wandb prompts during training
        )
        trainer = SFTTrainer(
            model=model, 
            processing_class=tokenizer, 
            train_dataset=split["train"],
            eval_dataset=split["test"], 
            args=args, 
        )
        trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        return trainer

if __name__ == "__main__":
    # --- PHASE 1: TRAINING ---
    print("Starting Training...")
    model, tokenizer = Brain().load()
    trainer = Trainer().run(model, tokenizer)
    
    # --- PHASE 2: CLEANUP ---
    # We must delete the 4-bit model to make room for the 16-bit merge
    print("Training complete. Clearing memory for merge...")
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # --- PHASE 3: MERGE ---
    print("Loading base model in FP16 for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        token=True
    )
    
    print("Applying adapters to base model...")
    # Load the adapters saved in OUT_DIR and merge them into the base weights
    model = PeftModel.from_pretrained(base_model, OUT_DIR)
    merged_model = model.merge_and_unload()

    # --- PHASE 4: SAVE ---
    print(f"Saving merged model to {MERGED_DIR}...")
    merged_model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    
    print("Process Complete! Download the 'merged_career_architect' folder.")
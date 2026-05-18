# 🧠 Career Architect SLM — Fine-Tuned Llama 3.1 8B

A fine-tuned Small Language Model (SLM) built on Meta's Llama 3.1 8B Instruct,
specialized for AI/ML career guidance — reflecting the grit of a professional
who transitioned from high-pressure service roles to AI research.

---

## 🚀 Run It Locally (5.1 GB RAM required)

```bash
ollama run career-architect
```

---

## 📌 Project Overview

| Component | Details |
|-----------|---------|
| Base Model | Meta Llama 3.1 8B Instruct |
| Method | QLoRA (4-bit quantization + LoRA) |
| LoRA Rank | r=8, alpha=16 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Training Epochs | 3 |
| Training Platform | Kaggle (free T4 GPU) |
| Deployment Format | GGUF Q8_0 via Ollama |
| RAM Required | 5.1 GB |

---

### Project Structure

```text
career-architect-slm/
│
├── train_model.py
│     └── Full training + merge pipeline (run on Kaggle)
│
├── test_model.py
│     └── Adapter validation script (run on Kaggle)
│
├── training_data.jsonl
│     └── Custom career/ML training dataset
│
├── adapter_config.json
│     └── LoRA configuration
│
├── Modelfile
│     └── Ollama deployment config
│
└── README.md


---

## ⚙️ How This Was Built — Step by Step

### Why Kaggle?
Training and testing were done entirely on **Kaggle** (free T4 GPU with 16GB VRAM)
because the local machine didn't have enough RAM to load a 8B model for training.
The scripts in this repo reflect the actual Kaggle workflow.

### Step 1 — Train on Kaggle
Upload `train_model.py` and `training_data.jsonl` to a Kaggle notebook and run it.

The script will:
1. Load Llama 3.1 8B Instruct in 4-bit (QLoRA)
2. Fine-tune with LoRA adapters on the career dataset
3. Merge the adapter into the base model (saved to `/tmp/`)
4. Output: merged model in HuggingFace format

### Step 2 — Validate on Kaggle
Run `test_model.py` on Kaggle to verify the fine-tuned adapter responds correctly
before converting and deploying.

### Step 3 — Convert to GGUF on Kaggle
Since the merged model (~16GB) is too large to download directly, we convert it
to GGUF format (Q8_0) using llama.cpp directly on Kaggle:

```python
import subprocess
subprocess.run(["git", "clone", "--branch", "b3447",
    "https://github.com/ggerganov/llama.cpp"], check=True)
subprocess.run(["pip", "install", "-r", "llama.cpp/requirements.txt", "-q"], check=True)
subprocess.run([
    "python", "llama.cpp/convert_hf_to_gguf.py",
    "/tmp/merged_career_architect",
    "--outtype", "q8_0",
    "--outfile", "/kaggle/working/career-architect.gguf"
], check=True)
```

This reduces the model from ~16GB to **8.5GB**.

### Step 4 — Upload to HuggingFace
The GGUF file is uploaded to HuggingFace directly from Kaggle:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="/kaggle/working/career-architect.gguf",
    path_in_repo="career-architect.gguf",
    repo_id="aksa2000/career-architect",
    repo_type="model"
)
```

### Step 5 — Deploy Locally with Ollama
Download the GGUF from HuggingFace, then:

```bash
ollama create career-architect -f Modelfile
ollama run career-architect
```

Ollama runs the model using only **5.1 GB of RAM**.

---

## 💬 Example Interaction

**User:** How does your background in high-pressure service environments
contribute to your grit as an ML Engineer?

**Career Architect:** My background in high-pressure service environments taught
me discipline and resilience under stress. These traits are essential for an ML
engineer, where experiments can fail, deadlines are tight, and continuous learning
is required. My experience has prepared me to handle the uncertainty and high
stakes of AI research with confidence and composure.

---

## 🔗 Model Weights
GGUF available on HuggingFace: [aksa2000/career-architect](https://huggingface.co/aksa2000/career-architect)

## 👤 Author
Built by [@akirem20](https://github.com/akirem20) — transitioning from
high-pressure service roles to ML Engineering.

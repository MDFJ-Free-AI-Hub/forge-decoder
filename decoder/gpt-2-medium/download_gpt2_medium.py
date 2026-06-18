"""
download_gpt_2_medium.py
────────────────────────────────────────────────
Downloads the weights and tokenizer for openai-community/gpt2-medium
using the HuggingFace Transformers library.

Model    : gpt-2-medium
HF Hub   : https://huggingface.co/openai-community/gpt2-medium
Size     : ~1.5 GB
Precision: float32

Requirements:
    pip install transformers accelerate torch

Usage:
    # Optional — required for gated models (Llama, Mistral, Falcon):
    export HF_TOKEN="hf_your_token_here"
    python download_gpt2_medium.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID  = "openai-community/gpt2-medium"
SAVE_DIR  = "./gpt-2-medium"
HF_TOKEN  = os.environ.get("HF_TOKEN")   # set this for gated models
TORCH_DTYPE = torch.float32

print(f"[{m.modelName}] Downloading tokenizer from HuggingFace Hub...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
)
tokenizer.save_pretrained(SAVE_DIR)
print(f"[{m.modelName}] Tokenizer saved → {SAVE_DIR}")

print(f"[{m.modelName}] Downloading model weights (~1.5 GB)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.save_pretrained(SAVE_DIR)
print(f"[{m.modelName}] Model weights saved → {SAVE_DIR}")
print("Done. You can now load the model offline with:")
print(f'  AutoModelForCausalLM.from_pretrained("{SAVE_DIR}", local_files_only=True)')

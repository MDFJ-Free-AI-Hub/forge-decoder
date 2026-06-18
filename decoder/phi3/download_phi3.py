"""
download_phi_3_mini_4k_instruct.py
────────────────────────────────────────────────
Downloads the weights and tokenizer for microsoft/Phi-3-mini-4k-instruct
using the HuggingFace Transformers library.

Model    : phi-3-mini-4k-instruct
HF Hub   : https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
Size     : ~7.6 GB
Precision: float16

Requirements:
    pip install transformers accelerate torch

Usage:
    # Optional — required for gated models (Llama, Mistral, Falcon):
    export HF_TOKEN="hf_your_token_here"
    python download_phi3.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID  = "microsoft/Phi-3-mini-4k-instruct"
SAVE_DIR  = "./phi-3-mini-4k-instruct"
HF_TOKEN  = os.environ.get("HF_TOKEN")   # set this for gated models
TORCH_DTYPE = torch.float16

print(f"[{m.modelName}] Downloading tokenizer from HuggingFace Hub...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
)
tokenizer.save_pretrained(SAVE_DIR)
print(f"[{m.modelName}] Tokenizer saved → {SAVE_DIR}")

print(f"[{m.modelName}] Downloading model weights (~7.6 GB)...")
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

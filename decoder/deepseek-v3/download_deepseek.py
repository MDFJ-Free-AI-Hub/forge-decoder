"""
download_deepseek_v3.py
────────────────────────────────────────────────
Downloads the weights and tokenizer for deepseek-ai/DeepSeek-V3
using the HuggingFace Transformers library.

Model    : deepseek-v3
HF Hub   : https://huggingface.co/deepseek-ai/DeepSeek-V3
Size     : ~641 GB (FP8) / ~1.3 TB (BF16)
Precision: bfloat16

Requirements:
    pip install transformers accelerate torch

Usage:
    # Optional — required for gated models (Llama, Mistral, Falcon):
    export HF_TOKEN="hf_your_token_here"
    python download_deepseek.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID  = "deepseek-ai/DeepSeek-V3"
SAVE_DIR  = "./deepseek-v3"
HF_TOKEN  = os.environ.get("HF_TOKEN")   # set this for gated models
TORCH_DTYPE = torch.bfloat16

print(f"[{m.modelName}] Downloading tokenizer from HuggingFace Hub...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
)
tokenizer.save_pretrained(SAVE_DIR)
print(f"[{m.modelName}] Tokenizer saved → {SAVE_DIR}")

print(f"[{m.modelName}] Downloading model weights (~641 GB (FP8) / ~1.3 TB (BF16))...")
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

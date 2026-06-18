"""
download_stable_diffusion_xl_base_1.0.py
────────────────────────────────────────────────
Downloads the pipeline weights for stabilityai/stable-diffusion-xl-base-1.0
using the HuggingFace Diffusers library.

Model    : stable-diffusion-xl-base-1.0
HF Hub   : https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
Size     : ~7 GB
Precision: float16

Requirements:
    pip install diffusers transformers accelerate safetensors torch

Usage:
    export HF_TOKEN="hf_your_token_here"   # optional for public models
    python download_sdxl.py
"""

import os
import torch
from diffusers import StableDiffusionXLPipeline

MODEL_ID  = "stabilityai/stable-diffusion-xl-base-1.0"
SAVE_DIR  = "./stable-diffusion-xl-base-1.0"
HF_TOKEN  = os.environ.get("HF_TOKEN")

print(f"[stable-diffusion-xl-base-1.0] Downloading SDXL pipeline (~7 GB)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    token=HF_TOKEN,
)
pipe.save_pretrained(SAVE_DIR)
print(f"[stable-diffusion-xl-base-1.0] Pipeline saved → {SAVE_DIR}")
print("Done. Load offline with:")
print(f'  StableDiffusionXLPipeline.from_pretrained("{SAVE_DIR}", torch_dtype=torch.float16)')

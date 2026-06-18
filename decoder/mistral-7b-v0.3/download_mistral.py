"""
download_mistral.py
══════════════════════════════════════════════════════════════
Descarga los pesos de mistralai/Mistral-7B-v0.3 desde HuggingFace
y los sube automáticamente a GitHub con git + LFS.

Modelo   : mistral-7b-v0.3
HF Hub   : https://huggingface.co/mistralai/Mistral-7B-v0.3
Tamaño   : ~14 GB
Repo GH  : https://github.com/MDFJ-Free-AI-Hub/forge-decoder
Carpeta  : decoder/mistral-7b-v0.3/

Requisitos:
    pip install transformers accelerate torch safetensors
    git lfs install   # instalar Git LFS una sola vez

Variables de entorno:
    GH_TOKEN    — GitHub token con permisos de escritura (obligatorio)
    HF_TOKEN    — HuggingFace token (OBLIGATORIO — modelo con acceso restringido)

Uso:
    export GH_TOKEN="ghp_tu_token_aqui"
    export HF_TOKEN="hf_tu_token_aqui"
    python download_mistral.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ── Configuración ─────────────────────────────────────────────────────────────
HF_MODEL_ID  = "mistralai/Mistral-7B-v0.3"
GH_ORG       = "MDFJ-Free-AI-Hub"
GH_REPO      = "forge-decoder"
REPO_FOLDER  = "decoder/mistral-7b-v0.3"   # carpeta dentro del repo
LOCAL_DIR    = os.path.join(tempfile.gettempdir(), "mistral-7b-v0.3")

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")
# Gated model — you must accept the license on HuggingFace first
# and set HF_TOKEN before running this script.

if not GH_TOKEN:
    sys.exit("ERROR: Necesitas GH_TOKEN con permisos de escritura al repo.")

# ── 1. Descargar pesos desde HuggingFace ──────────────────────────────────────
print(f"[1/4] Descargando {HF_MODEL_ID} → {LOCAL_DIR}  (~14 GB)")
os.makedirs(LOCAL_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
)
tokenizer.save_pretrained(LOCAL_DIR)

model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.save_pretrained(LOCAL_DIR)

print(f"[1/4] Descarga completada.")

# ── 2. Clonar el repo de GitHub ────────────────────────────────────────────────
clone_url = f"https://oauth2:{GH_TOKEN}@github.com/{GH_ORG}/{GH_REPO}.git"
repo_dir  = os.path.join(tempfile.gettempdir(), GH_REPO)

print(f"[2/4] Clonando {GH_ORG}/{GH_REPO}...")
if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)
subprocess.run(["git", "clone", "--depth=1", clone_url, repo_dir], check=True)

# Configurar Git LFS en el repo clonado
os.chdir(repo_dir)
subprocess.run(["git", "lfs", "install"], check=True)
subprocess.run(["git", "lfs", "track", "*.safetensors", "*.bin", "*.pt", "*.ot", "*.gguf"], check=True)
subprocess.run(["git", "add", ".gitattributes"], check=False)

# ── 3. Copiar los archivos de pesos al repo ────────────────────────────────────
dest = os.path.join(repo_dir, REPO_FOLDER)
os.makedirs(dest, exist_ok=True)

print(f"[3/4] Copiando archivos al repo → {REPO_FOLDER}/")
for fname in os.listdir(LOCAL_DIR):
    src  = os.path.join(LOCAL_DIR, fname)
    dst  = os.path.join(dest, fname)
    if os.path.isfile(src):
        size_mb = os.path.getsize(src) / 1024 / 1024
        print(f"       {fname}  ({size_mb:.1f} MB)")
        shutil.copy2(src, dst)

# ── 4. Commit y push ───────────────────────────────────────────────────────────
print(f"[4/4] Subiendo a GitHub...")
subprocess.run(["git", "config", "user.email", "aiva-forge@noreply.github.com"], check=True)
subprocess.run(["git", "config", "user.name",  "Aiva AI Forge"], check=True)
subprocess.run(["git", "add", REPO_FOLDER], check=True)
subprocess.run(["git", "commit", "-m", f"feat: add weights for mistral-7b-v0.3 ({m.approxSize})"], check=True)
subprocess.run(["git", "push"], check=True)

print(f"\n✅  Pesos de mistral-7b-v0.3 subidos correctamente.")
print(f"    https://github.com/{GH_ORG}/{GH_REPO}/tree/main/{m.folder}")

# Limpieza
os.chdir("/")
shutil.rmtree(LOCAL_DIR, ignore_errors=True)
shutil.rmtree(repo_dir,  ignore_errors=True)
print("   Archivos temporales eliminados.")

"""Fail installation before the launcher marks a partially installed app ready."""
from pathlib import Path
import platform

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise SystemExit("IDOL requires PyTorch >=2.5; native Intel macOS has no compatible wheel.")

import fastapi
import uvicorn
import torch
from idol import IdolPipeline
from diffusers.models import AutoencoderKLFlux2
from transformers import DINOv3ViTModel

assert hasattr(IdolPipeline, "from_pretrained")
root = Path(__file__).resolve().parent
assert (root / "static/vendor/gaussian-splats-3d.module.js").is_file()
assert (root / "static/vendor/three.module.js").is_file()
Path(__file__).with_name(".installed").write_text("ready\n")
print(f"IDOL Local installed. PyTorch {torch.__version__}; FastAPI {fastapi.__version__}.")

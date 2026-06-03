"""
modal_config.py — shared Modal app, volumes, and base images.

All pipeline and training scripts import from here to share the same
Modal app name, volumes, and pre-built container images.

Usage:
    from modal_config import app, models_volume, data_volume, pipeline_image, training_image
"""

import modal
import os
from pathlib import Path

# ── Load .env so local modal commands pick up API keys ───────────────────────
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _ln in _env.read_text().splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _v = _ln.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── App ──────────────────────────────────────────────────────────────────────
app = modal.App("4drecon")

# ── Persistent volumes ───────────────────────────────────────────────────────
# models_volume: large model weights (DepthPro, SAM2, CogVideoX-5b-I2V, etc.)
# data_volume:   per-run scene data, outputs, and checkpoints
models_volume = modal.Volume.from_name("4drecon-models", create_if_missing=True)
data_volume   = modal.Volume.from_name("4drecon-data",   create_if_missing=True)

# ── Mount paths (inside container) ───────────────────────────────────────────
MODELS_PATH = "/models"
DATA_PATH   = "/data"

# ── Base CUDA image ───────────────────────────────────────────────────────────
_base = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install([
        "git", "git-lfs", "wget", "curl", "ffmpeg",
        "libgl1-mesa-glx", "libglib2.0-0",          # OpenCV deps
        "build-essential", "cmake",                   # for CUDA extension builds
    ])
    .pip_install("torch==2.3.0+cu121", "torchvision==0.18.0+cu121",
                 extra_index_url="https://download.pytorch.org/whl/cu121")
)

# ── Pipeline image (A10G tasks: depth, pose, segmentation) ───────────────────
pipeline_image = (
    _base
    .pip_install([
        "diffusers>=0.30.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.3",
        "opencv-python>=4.9.0",
        "imageio>=2.34.0",
        "imageio-ffmpeg>=0.5.0",
        "Pillow>=10.3.0",
        "plyfile>=1.0.3",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.66.0",
        "ffmpeg-python>=0.2.0",
        "rich>=13.7.0",
        "open3d>=0.18.0",
        "huggingface_hub>=0.22.0",
    ])
    # Apple Depth-Pro — not on PyPI, install from GitHub
    .run_commands(
        "git clone --depth 1 https://github.com/apple/ml-depth-pro /opt/ml-depth-pro",
        "pip install -e /opt/ml-depth-pro",
    )
    # Clone DUSt3R — importable via PYTHONPATH (no setup.py, so no pip install -e)
    # The base image already has torch, torchvision, scipy, einops, matplotlib,
    # tqdm, opencv-python, huggingface-hub, numpy<2, safetensors>=0.8.0rc0.
    # The only DUSt3R dep NOT in the base is `roma` (rotation math library).
    # Installing just roma avoids the pip resolver backtracking that occurs when
    # trying to reconcile the full requirements.txt against already-pinned packages.
    .run_commands(
        "git clone --depth 1 https://github.com/naver/dust3r /opt/dust3r",
        "pip install --no-deps roma",
    )
    # Clone SAM2 and install
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/sam2 /opt/sam2",
        "pip install -e /opt/sam2",
    )
    .env({"PYTHONPATH": "/opt/dust3r:/opt/sam2:$PYTHONPATH"})
    # Make modal_config importable in the container
    .add_local_python_source("modal_config")
)

# ── Training image (A100 tasks: 3DGS, Deformable-3DGS, CogVideoX) ────────────
training_image = (
    _base
    .pip_install([
        "diffusers>=0.30.0",
        "transformers>=4.40.0",
        "peft>=0.11.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.3",
        "bitsandbytes>=0.43.0",
        "wandb>=0.17.0",
        "opencv-python>=4.9.0",
        "imageio>=2.34.0",
        "imageio-ffmpeg>=0.5.0",
        "Pillow>=10.3.0",
        "plyfile>=1.0.3",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "lpips>=0.1.4",
        "scikit-image>=0.22.0",
    ])
    # Clone gaussian-splatting and build CUDA extensions
    .run_commands(
        "git clone --recursive --depth 1 "
        "https://github.com/graphdeco-inria/gaussian-splatting /opt/gaussian-splatting",
        "pip install -e /opt/gaussian-splatting/submodules/diff-gaussian-rasterization",
        "pip install -e /opt/gaussian-splatting/submodules/simple-knn",
    )
    # Clone Deformable-3D-Gaussians
    .run_commands(
        "git clone --depth 1 "
        "https://github.com/ingra14m/Deformable-3D-Gaussians /opt/deformable-3dgs",
        "pip install -r /opt/deformable-3dgs/requirements.txt || true",
    )
    .env({
        "PYTHONPATH": (
            "/opt/gaussian-splatting:"
            "/opt/deformable-3dgs:"
            "$PYTHONPATH"
        )
    })
    # Make modal_config importable in the container
    .add_local_python_source("modal_config")
)

# ── Convenience: volume mapping dict used by all functions ────────────────────
VOLUME_MAP = {
    MODELS_PATH: models_volume,
    DATA_PATH:   data_volume,
}

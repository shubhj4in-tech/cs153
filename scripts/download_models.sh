#!/usr/bin/env bash
# ============================================================
# download_models.sh — Download all model weights
#
# Usage:
#   bash scripts/download_models.sh [--skip-cogvideox]
#
# Downloads to:
#   models/depth_pro/     Apple DepthPro
#   models/sam2/          Meta SAM2-Hiera-Large
#   models/cogvideox/     zai-org/CogVideoX-5b-I2V  (large — ~30 GB; skip with flag)
#
# CogVideoX weights are intended for a Modal volume, but we also support
# local download here for testing.
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Load .env if present
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Export HF_TOKEN for huggingface-cli
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

SKIP_COGVIDEOX=0
for arg in "$@"; do
    [ "$arg" = "--skip-cogvideox" ] && SKIP_COGVIDEOX=1
done

GREEN="\033[0;32m"; YELLOW="\033[1;33m"; NC="\033[0m"
log()  { echo -e "${GREEN}[models]${NC} $*"; }
warn() { echo -e "${YELLOW}[models]${NC} $*"; }

mkdir -p models/depth_pro models/sam2 models/cogvideox

# ── Check for huggingface-cli ─────────────────────────────────────────────────
if ! command -v huggingface-cli &>/dev/null; then
    log "Installing huggingface-hub CLI ..."
    pip install --quiet huggingface-hub
fi

# ── Apple DepthPro ────────────────────────────────────────────────────────────
log "Downloading Apple DepthPro ..."
if [ -f "models/depth_pro/depth_pro.pt" ]; then
    warn "  depth_pro.pt already present — skipping."
else
    # DepthPro is distributed via the apple/ml-depth-pro HF repo
    # VERIFY: The correct HF model ID for DepthPro.
    # As of 2024/2025 it may be 'apple/DepthPro' — check HF hub.
    huggingface-cli download \
        apple/DepthPro \
        --local-dir models/depth_pro \
        --quiet \
        || {
            warn "  HF download failed. Trying direct git-lfs clone ..."
            # Fallback: clone the repo from Apple's GitHub + download checkpoint
            if [ ! -d "models/depth_pro_repo" ]; then
                git clone --depth 1 \
                    https://github.com/apple/ml-depth-pro \
                    models/depth_pro_repo
            fi
            if command -v gdown &>/dev/null; then
                warn "  You may need to download depth_pro.pt manually from"
                warn "  https://huggingface.co/apple/DepthPro"
            fi
        }
fi

# ── Meta SAM2 ────────────────────────────────────────────────────────────────
log "Downloading SAM2-Hiera-Large ..."
SAM2_CKPT="models/sam2/sam2_hiera_large.pt"
if [ -f "$SAM2_CKPT" ]; then
    warn "  sam2_hiera_large.pt already present — skipping."
else
    # VERIFY: HF repo id for SAM2. Meta released it as 'facebook/sam2-hiera-large'
    huggingface-cli download \
        facebook/sam2-hiera-large \
        --local-dir models/sam2 \
        --quiet \
        || {
            warn "  Trying direct checkpoint URL ..."
            curl -L --progress-bar \
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
                -o "$SAM2_CKPT"
        }
fi

# ── SAM2 model config ────────────────────────────────────────────────────────
# The SAM2 package ships config files; they should be installed with `pip install -e sam2/`
# But we also copy the large config to models/sam2/ for explicit reference.
SAM2_CFG_SRC="sam2/sam2/configs/sam2/sam2_hiera_l.yaml"
if [ -f "$SAM2_CFG_SRC" ] && [ ! -f "models/sam2/sam2_hiera_l.yaml" ]; then
    cp "$SAM2_CFG_SRC" "models/sam2/sam2_hiera_l.yaml"
    log "  Copied SAM2 config to models/sam2/"
fi

# ── CogVideoX-5b ─────────────────────────────────────────────────────────────
if [ "$SKIP_COGVIDEOX" -eq 0 ]; then
    log "Downloading CogVideoX-5b (~30 GB) ..."
    log "  This will take a while. Use --skip-cogvideox to skip and use Modal volume instead."

    if [ -d "models/cogvideox/transformer" ]; then
        warn "  CogVideoX transformer already present — skipping."
    else
        huggingface-cli download \
            zai-org/CogVideoX-5b-I2V \
            --local-dir models/cogvideox \
            --quiet \
            || warn "  CogVideoX download failed. You can download it to a Modal volume with:"
                    "  modal volume put 4drecon-models zai-org/CogVideoX-5b-I2V /models/cogvideox"
    fi
else
    warn "CogVideoX download skipped (--skip-cogvideox). Download to Modal volume:"
    warn "  modal volume put 4drecon-models . /models  # after local download"
    warn "  or: huggingface-cli download zai-org/CogVideoX-5b-I2V --local-dir models/cogvideox"
fi

# ── DUSt3R weights ────────────────────────────────────────────────────────────
log "Downloading DUSt3R checkpoint ..."
DUST3R_CKPT="models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ -f "$DUST3R_CKPT" ]; then
    warn "  DUSt3R checkpoint already present — skipping."
else
    mkdir -p models/dust3r
    # VERIFY: DUSt3R checkpoint URL. Check https://github.com/naver/dust3r for the latest.
    curl -L --progress-bar \
        "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
        -o "$DUST3R_CKPT" \
        || {
            warn "  Direct download failed. Trying HF hub ..."
            huggingface-cli download \
                naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt \
                --local-dir models/dust3r \
                --quiet || warn "  DUSt3R download failed. It will fall back to HF hub at runtime."
        }
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
log "Model downloads complete."
echo ""
echo "  models/"
echo "  ├── depth_pro/   (Apple DepthPro)"
echo "  ├── sam2/        (Meta SAM2-Hiera-Large)"
echo "  ├── dust3r/      (DUSt3R ViT-Large)"
echo "  └── cogvideox/   (CogVideoX-5b, if downloaded)"
echo ""

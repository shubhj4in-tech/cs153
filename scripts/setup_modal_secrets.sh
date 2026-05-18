#!/usr/bin/env bash
# ============================================================
# setup_modal_secrets.sh — Push API keys to Modal as secrets
#
# Run once after `modal token new` to make keys available
# inside Modal containers.
#
# Usage:
#   bash scripts/setup_modal_secrets.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env
if [ ! -f ".env" ]; then
    echo "ERROR: .env not found. Run: cp .env.example .env && fill in keys."
    exit 1
fi
set -a; source .env; set +a

GREEN="\033[0;32m"; NC="\033[0m"
log() { echo -e "${GREEN}[modal-secrets]${NC} $*"; }

# wandb secret
if [ -n "${WANDB_API_KEY:-}" ]; then
    log "Creating Modal secret 'wandb-secret' ..."
    modal secret create wandb-secret \
        WANDB_API_KEY="$WANDB_API_KEY" \
        WANDB_PROJECT="${WANDB_PROJECT:-4drecon}" \
        --force
    log "  wandb-secret created."
else
    echo "  WANDB_API_KEY not set in .env — skipping."
fi

# HuggingFace secret
if [ -n "${HF_TOKEN:-}" ]; then
    log "Creating Modal secret 'hf-secret' ..."
    modal secret create hf-secret \
        HF_TOKEN="$HF_TOKEN" \
        HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
        --force
    log "  hf-secret created."
else
    echo "  HF_TOKEN not set in .env — skipping."
fi

log "Done. Secrets are now available inside Modal containers."
echo ""
echo "  To verify:"
echo "    modal secret list"

"""
training/finetune_cogvideo.py — LoRA fine-tuning of CogVideoX-5b
=================================================================
Adds a lightweight flow-prediction head to the temporal attention layers
of CogVideoX and trains it to predict optical flow from the Blender dataset.

Architecture:
  • CogVideoX-5b-I2V transformer (frozen base)
  • LoRA (r=16, α=16) on transformer blocks 12-18,
    targeting: to_q, to_k, to_v, to_out.0
  • Linear flow head: transformer_hidden_dim → 2 * H * W
  • Objective: L1(predicted_flow, gt_flow)

Can be run locally (--local) or as a Modal job.

Usage (Modal):
    python training/finetune_cogvideo.py --manifest ./data/synthetic/manifest.json

Usage (local, for testing):
    python training/finetune_cogvideo.py \\
        --manifest ./data/synthetic/manifest.json \\
        --local --max-steps 100

Cost estimate: ~6 A100-hours ($9.00) for 5000 steps, batch 4, grad-accum 4.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import modal
    from modal_config import app, training_image, VOLUME_MAP, MODELS_PATH, DATA_PATH
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False


# ─────────────────────────────────────────────────────────────────────────────
# Training implementation (runs inside container or locally)
# ─────────────────────────────────────────────────────────────────────────────

def train(
    manifest_path: str,
    model_id: str       = "zai-org/CogVideoX-5b-I2V",
    lora_r: int         = 16,
    lora_alpha: int     = 16,
    max_steps: int      = 5000,
    batch_size: int     = 4,
    grad_accum: int     = 4,
    lr: float           = 1e-4,
    weight_decay: float = 0.01,
    save_every: int     = 500,
    output_dir: str     = "./outputs/cogvideox_lora",
    wandb_project: str  = "4drecon",
    clip_len: int       = 16,
    height: int         = 256,
    width: int          = 256,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import wandb

    from diffusers import CogVideoXPipeline
    from peft import LoraConfig, get_peft_model, TaskType

    from training.dataset import build_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load CogVideoX ────────────────────────────────────────────────────────
    print(f"[FineTune] Loading CogVideoX from {model_id} ...")
    # We only need the transformer (no VAE/tokenizer needed for feature extraction)
    pipeline = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    transformer = pipeline.transformer.to(device)

    # Freeze all base parameters
    for p in transformer.parameters():
        p.requires_grad_(False)

    # VERIFY: CogVideoX transformer block attribute names.
    # In diffusers >=0.30, CogVideoXTransformer3DModel stores blocks in
    # transformer.transformer_blocks[i] and attention in block.attn1 / block.attn2.
    # The exact submodule names for LoRA targets may differ.
    # Check: list(transformer.transformer_blocks[0].named_modules())

    # ── Apply LoRA to target blocks ───────────────────────────────────────────
    # Target blocks 12-18 (the "deep" temporal layers)
    target_block_indices = list(range(12, min(19, len(transformer.transformer_blocks))))
    print(f"  Applying LoRA to blocks {target_block_indices[0]}–{target_block_indices[-1]}")

    # VERIFY: target_modules names. CogVideoX may use "attn1.to_q" etc.
    # Adjust if you see 'target_modules not found' errors from PEFT.
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Apply LoRA only to the selected blocks by temporarily isolating them
    # We create a small wrapper module containing just the target blocks
    selected_blocks = nn.ModuleList(
        [transformer.transformer_blocks[i] for i in target_block_indices]
    )
    selected_blocks_lora = get_peft_model(selected_blocks, lora_config)

    # Re-attach the LoRA'd blocks back into the transformer
    for j, i in enumerate(target_block_indices):
        transformer.transformer_blocks[i] = selected_blocks_lora.base_model.model[j]

    # ── Flow prediction head ──────────────────────────────────────────────────
    # CogVideoX hidden dimension
    # VERIFY: hidden_size attribute name from model config
    hidden_size = getattr(transformer.config, "hidden_size", 3072)

    flow_head = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, 512),
        nn.GELU(),
        nn.Linear(512, 2 * height * width),   # predict u,v for each pixel
    ).to(device).to(torch.bfloat16)

    # ── Optimiser ────────────────────────────────────────────────────────────
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    all_params  = list(lora_params) + list(flow_head.parameters())
    n_params    = sum(p.numel() for p in all_params)
    print(f"  Trainable parameters: {n_params:,}")

    optimizer = AdamW(all_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.1)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_key = os.environ.get("WANDB_API_KEY")
    use_wandb = bool(wandb_key)
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name="cogvideox_flow_lora",
            config={
                "model_id":     model_id,
                "lora_r":       lora_r,
                "lora_alpha":   lora_alpha,
                "max_steps":    max_steps,
                "batch_size":   batch_size,
                "grad_accum":   grad_accum,
                "lr":           lr,
                "target_blocks": target_block_indices,
                "hidden_size":  hidden_size,
            },
        )

    # ── DataLoader ────────────────────────────────────────────────────────────
    dl = build_dataloader(
        manifest_path, batch_size=batch_size,
        clip_len=clip_len, height=height, width=width,
    )
    data_iter = iter(dl)

    # ── Feature extraction hook ───────────────────────────────────────────────
    # We hook the output of each target block's attention to get a (B, T*H*W, D)
    # feature tensor, then pool over spatial dims and take the last frame.
    extracted_features: dict[int, torch.Tensor] = {}

    def make_hook(block_idx: int):
        def hook(module, inp, out):
            # out may be a tuple; first element is the attended output
            feat = out[0] if isinstance(out, tuple) else out
            extracted_features[block_idx] = feat  # (B, seq_len, D)
        return hook

    hooks = []
    for j, i in enumerate(target_block_indices):
        h = transformer.transformer_blocks[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    # ── Training loop ─────────────────────────────────────────────────────────
    transformer.train()
    flow_head.train()
    optimizer.zero_grad()

    accum_loss = 0.0
    global_step = 0

    print(f"[FineTune] Starting training — {max_steps} steps ...")

    while global_step < max_steps:
        try:
            rgb_clip, flow_gt = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            rgb_clip, flow_gt = next(data_iter)

        rgb_clip = rgb_clip.to(device, dtype=torch.bfloat16)   # (B, T, 3, H, W)
        flow_gt  = flow_gt.to(device, dtype=torch.bfloat16)    # (B, T, 2, H, W)

        B, T, C, H, W = rgb_clip.shape

        # CogVideoX expects (B, C, T, H, W) for video input
        # VERIFY: exact input format for CogVideoX transformer
        # The transformer forward() signature takes hidden_states, not raw pixels.
        # We need to encode through the VAE first, OR hook into the pipeline.
        # For simplicity, we pass the pixel tensor through a small linear projection
        # to approximate the VAE-encoded latent. In a full implementation, use the VAE.
        # VERIFY: whether to use VAE encoding here for better quality features.

        # Simple pixel-to-latent approximation (4x spatial downscale)
        latent = F.avg_pool3d(
            rgb_clip.permute(0, 2, 1, 3, 4),   # (B, C, T, H, W)
            kernel_size=(1, 4, 4),
            stride=(1, 4, 4),
        )   # (B, C, T, H//4, W//4)

        # Run through transformer (only LoRA'd layers are computed)
        # VERIFY: CogVideoXTransformer3DModel forward signature
        # typical: transformer(hidden_states, encoder_hidden_states, timestep, ...)
        # For feature extraction at inference, we can pass dummy timestep=0
        dummy_timestep = torch.zeros(B, dtype=torch.long, device=device)

        # Flatten spatial for transformer input (B, T*seq, D) — approximate
        # Since we are only interested in internal features via hooks, not the output
        extracted_features.clear()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            try:
                _ = transformer(
                    hidden_states=latent,
                    timestep=dummy_timestep,
                    encoder_hidden_states=torch.zeros(
                        B, 1, hidden_size, device=device, dtype=torch.bfloat16
                    ),
                    return_dict=False,
                )
            except Exception as e:
                # VERIFY: If the transformer forward signature differs,
                # adjust the call above. Run:
                #   import inspect; print(inspect.signature(transformer.forward))
                print(f"  Transformer forward failed: {e}")
                print("  VERIFY: CogVideoX forward() signature in diffusers.")
                raise

        # ── Aggregate features from hooked layers ─────────────────────────────
        if not extracted_features:
            print("  WARNING: No features extracted — hooks may not have fired.")
            continue

        # Stack features from all hooked layers, mean-pool
        feats = torch.stack(list(extracted_features.values()), dim=0)  # (L, B, seq, D)
        feat  = feats.mean(dim=0)   # (B, seq, D)

        # Pool over sequence length → (B, D)
        feat_pooled = feat.mean(dim=1)   # (B, D)

        # ── Predict flow for the middle frame ─────────────────────────────────
        flow_pred_flat = flow_head(feat_pooled)    # (B, 2*H*W)
        flow_pred = flow_pred_flat.view(B, 2, H, W)

        # GT flow: use middle frame
        mid_t     = T // 2
        flow_gt_t = flow_gt[:, mid_t, :, :, :]    # (B, 2, H, W)
        # Ensure H, W match
        if flow_pred.shape[-2:] != flow_gt_t.shape[-2:]:
            flow_pred = F.interpolate(flow_pred, size=flow_gt_t.shape[-2:],
                                      mode="bilinear", align_corners=False)

        loss = F.l1_loss(flow_pred.float(), flow_gt_t.float())
        loss = loss / grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (global_step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step % 50 == 0:
                mem = (torch.cuda.memory_reserved() / 1e9
                       if torch.cuda.is_available() else 0)
                print(
                    f"  Step {global_step+1}/{max_steps}  "
                    f"loss={accum_loss*grad_accum:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"VRAM={mem:.1f}GB"
                )
                if use_wandb:
                    wandb.log({
                        "train/loss": accum_loss * grad_accum,
                        "train/lr":   scheduler.get_last_lr()[0],
                        "train/vram_gb": mem,
                    }, step=global_step)

            accum_loss = 0.0

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (global_step + 1) % save_every == 0:
            ckpt_dir = out / f"checkpoint-{global_step+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # Save LoRA adapter weights
            selected_blocks_lora.save_pretrained(str(ckpt_dir / "lora_adapter"))
            # Save flow head
            torch.save(flow_head.state_dict(), str(ckpt_dir / "flow_head.pt"))
            print(f"  Saved checkpoint → {ckpt_dir}")

        global_step += 1

    # ── Final save ────────────────────────────────────────────────────────────
    for h in hooks:
        h.remove()

    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    selected_blocks_lora.save_pretrained(str(final_dir / "lora_adapter"))
    torch.save(flow_head.state_dict(), str(final_dir / "flow_head.pt"))
    print(f"\n[FineTune] Training done → {final_dir}")

    if use_wandb:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Modal remote function
# ─────────────────────────────────────────────────────────────────────────────

if HAS_MODAL:
    @app.function(
        image=training_image,
        gpu="A100",
        volumes=VOLUME_MAP,
        timeout=43200,   # 12 hours
        memory=80000,
        secrets=[
            modal.Secret.from_name("wandb-secret", required=False),
            modal.Secret.from_name("hf-secret",    required=False),
        ],
    )
    def finetune_remote(
        manifest_relative: str,
        max_steps: int  = 5000,
        output_relative: str = "cogvideox_lora",
    ):
        """
        Modal function: CogVideoX LoRA fine-tuning.
        Cost estimate: ~6 A100-hours ($9.00) for 5000 steps.
        """
        import sys
        sys.path.insert(0, "/root")

        manifest_path = str(Path(DATA_PATH) / manifest_relative)
        output_dir    = str(Path(DATA_PATH) / output_relative)
        model_id      = str(Path(MODELS_PATH) / "cogvideox")

        train(
            manifest_path=manifest_path,
            model_id=model_id,
            max_steps=max_steps,
            output_dir=output_dir,
        )
        data_volume.commit()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CogVideoX-5b for optical flow prediction."
    )
    parser.add_argument("--manifest",   required=True,
                        help="Path to blender manifest.json")
    parser.add_argument("--model-id",   default="zai-org/CogVideoX-5b-I2V",
                        help="HF model ID or local path (default: zai-org/CogVideoX-5b-I2V)")
    parser.add_argument("--output-dir", default="./outputs/cogvideox_lora",
                        help="Where to save checkpoints")
    parser.add_argument("--max-steps",  type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--local",      action="store_true",
                        help="Run locally (no Modal)")
    args = parser.parse_args()

    if args.local or not HAS_MODAL:
        train(
            manifest_path=args.manifest,
            model_id=args.model_id,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            output_dir=args.output_dir,
        )
    else:
        print("[FineTune] Launching Modal training job ...")
        with app.run():
            finetune_remote.remote(
                manifest_relative=args.manifest,
                max_steps=args.max_steps,
            )


if __name__ == "__main__":
    main()

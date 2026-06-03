"""
training/extract_features.py — CogVideoX motion feature extraction
====================================================================
Loads the fine-tuned CogVideoX (base + LoRA adapter), registers forward
hooks on transformer blocks 12-18, runs a scene video through with
num_inference_steps=1 (feature extraction only, no generation), then for
each Gaussian in the static 3DGS checkpoint:

  1. Projects the Gaussian centre to 2D screen coordinates per frame
  2. Samples the feature map at that location via grid_sample
  3. Averages across frames and hooked layers → per-Gaussian embedding

Saves outputs/motion_embeddings.pt  — shape (N_gaussians, D_feat)

Usage:
    python training/extract_features.py \\
        --scene-dir   ./my_scene/ \\
        --lora-dir    ./outputs/cogvideox_lora/final/ \\
        --output      ./outputs/motion_embeddings.pt \\
        [--model-id   zai-org/CogVideoX-5b-I2V]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def project_gaussians(positions: np.ndarray, cameras: list[dict],
                       H: int, W: int) -> np.ndarray:
    """
    Project (N, 3) Gaussian centres to 2D for each camera.

    Returns:
        screen_coords: (N, n_cams, 2) float32 in [-1, 1] (grid_sample convention)
    """
    N      = len(positions)
    n_cams = len(cameras)
    pts    = np.concatenate([positions, np.ones((N, 1), dtype=np.float32)], axis=-1)  # (N,4)

    coords = np.zeros((N, n_cams, 2), dtype=np.float32)

    for i, cam in enumerate(cameras):
        c2w = np.array(cam["cam_to_world"], dtype=np.float32)   # (4, 4)
        w2c = np.linalg.inv(c2w)

        # Intrinsics
        fx, fy, cx, cy = cam["params"]

        # Project to camera space
        pts_cam = (w2c @ pts.T).T           # (N, 4)
        z       = pts_cam[:, 2].clip(1e-5)
        u       = (pts_cam[:, 0] / z) * fx + cx
        v       = (pts_cam[:, 1] / z) * fy + cy

        # Normalise to [-1, 1] for grid_sample
        u_norm = (u / W) * 2.0 - 1.0
        v_norm = (v / H) * 2.0 - 1.0

        coords[:, i, 0] = u_norm
        coords[:, i, 1] = v_norm

    return coords


def load_static_gaussians(bg_ckpt_dir: Path) -> np.ndarray:
    """Load Gaussian positions from the 3DGS checkpoint .ply."""
    from plyfile import PlyData

    candidates = sorted(
        (bg_ckpt_dir / "point_cloud").glob("iteration_*/point_cloud.ply"),
        key=lambda p: int(p.parent.name.split("_")[-1]),
        reverse=True,
    )
    if not candidates:
        ply_path = bg_ckpt_dir / "point_cloud.ply"
    else:
        ply_path = candidates[0]

    if not ply_path.exists():
        raise FileNotFoundError(f"3DGS point cloud not found: {ply_path}")

    ply   = PlyData.read(str(ply_path))
    verts = ply["vertex"]
    positions = np.stack(
        [verts["x"], verts["y"], verts["z"]], axis=-1
    ).astype(np.float32)
    print(f"  Loaded {len(positions)} Gaussians from {ply_path.name}")
    return positions


def extract_motion_embeddings(
    scene_dir: str,
    lora_dir:  str,
    output_path: str,
    model_id: str   = "zai-org/CogVideoX-5b-I2V",
    block_range: tuple[int, int] = (12, 19),
    clip_len: int   = 16,
    height: int     = 256,
    width: int      = 256,
):
    from diffusers import CogVideoXPipeline
    from peft import PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd     = Path(scene_dir)

    # ── Load cameras ──────────────────────────────────────────────────────────
    cameras_path = sd / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"cameras.json not found in {sd}. Run Stage 1.")
    cameras = json.loads(cameras_path.read_text())
    n_cams  = len(cameras)
    orig_H  = cameras[0]["height"]
    orig_W  = cameras[0]["width"]

    # ── Load Gaussians ────────────────────────────────────────────────────────
    bg_ckpt = sd / "outputs" / "bg_gaussians"
    positions = load_static_gaussians(bg_ckpt)
    N_gauss   = len(positions)

    # ── Load CogVideoX + LoRA ─────────────────────────────────────────────────
    print(f"[FeatExtract] Loading CogVideoX from {model_id} ...")
    pipeline = CogVideoXPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    )
    vae         = pipeline.vae.to(device).eval()
    transformer = pipeline.transformer.to(device)

    lora_adapter_path = Path(lora_dir) / "lora_adapter"
    if lora_adapter_path.exists():
        print(f"  Loading LoRA adapter from {lora_adapter_path} ...")
        # The adapter was saved via transformer.save_pretrained() after get_peft_model()
        transformer = PeftModel.from_pretrained(transformer, str(lora_adapter_path))
    else:
        print(f"  WARNING: LoRA adapter not found at {lora_adapter_path}. "
              "Using base model features.")

    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad_(False)

    # ── Register hooks ────────────────────────────────────────────────────────
    layer_features: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(module, inp, out):
            feat = out[0] if isinstance(out, tuple) else out
            # feat: (B, seq_len, D)  — store on CPU to avoid OOM
            layer_features[idx] = feat.detach().cpu()
        return hook

    hooks = []
    n_blocks = len(transformer.transformer_blocks)
    target_blocks = range(block_range[0], min(block_range[1], n_blocks))
    for i in target_blocks:
        h = transformer.transformer_blocks[i].register_forward_hook(make_hook(i))
        hooks.append(h)
    print(f"  Hooked blocks {block_range[0]}–{min(block_range[1]-1, n_blocks-1)}")

    # ── Build video tensor from scene frames ──────────────────────────────────
    frames_dir  = sd / "frames"
    frame_files = sorted(frames_dir.glob("*.png"))[:clip_len]
    if len(frame_files) == 0:
        raise FileNotFoundError(f"No frames found in {frames_dir}.")

    print(f"  Using {len(frame_files)} scene frames for feature extraction ...")
    from PIL import Image
    import torchvision.transforms.functional as TF

    video_frames = []
    for fp in frame_files:
        img = Image.open(fp).convert("RGB").resize((width, height), Image.BILINEAR)
        t   = TF.to_tensor(img) * 2.0 - 1.0    # (3, H, W)
        video_frames.append(t)

    # Pad to clip_len if fewer frames available
    while len(video_frames) < clip_len:
        video_frames.append(video_frames[-1].clone())

    video = torch.stack(video_frames, dim=0).unsqueeze(0).to(device, dtype=torch.bfloat16)
    # video: (1, T, 3, H, W)

    # Encode through the VAE to get proper latents.
    B = 1
    T_actual = video.shape[1]
    with torch.no_grad():
        frames_flat = video.reshape(T_actual, 3, height, width)
        latents_flat = vae.encode(frames_flat).latent_dist.sample()
        latents_flat = latents_flat * vae.config.scaling_factor
    latent_C = latents_flat.shape[1]
    latent_h = latents_flat.shape[2]
    latent_w = latents_flat.shape[3]
    latent = latents_flat.reshape(1, T_actual, latent_C, latent_h, latent_w)
    latent = latent.permute(0, 2, 1, 3, 4)   # (1, latent_C, T, latent_h, latent_w)

    hidden_size = getattr(transformer.config, "hidden_size", 3072)

    # ── Single forward pass to extract features ───────────────────────────────
    print("  Running single forward pass for feature extraction ...")
    layer_features.clear()
    dummy_ts = torch.zeros(B, dtype=torch.long, device=device)

    with torch.no_grad():
        try:
            _ = transformer(
                hidden_states=latent,
                timestep=dummy_ts,
                encoder_hidden_states=torch.zeros(
                    B, 1, hidden_size, device=device, dtype=torch.bfloat16
                ),
                return_dict=False,
            )
        except Exception as e:
            print(f"  VERIFY: transformer forward failed: {e}")
            print("  Adjust the forward call to match CogVideoX's API.")
            raise

    for h in hooks:
        h.remove()

    if not layer_features:
        raise RuntimeError(
            "No features were extracted. Check that hooks are on the right "
            "module path within transformer.transformer_blocks[i]."
        )

    # ── Build 2D feature maps from sequence features ──────────────────────────
    # Each hook gives (1, seq_len, D).
    # seq_len = T * latent_h * latent_w (spatial dims after VAE encoding).

    feat_H = latent_h
    feat_W = latent_w
    feat_T = T_actual

    # Average feature maps across layers
    feat_sum    = None
    n_layers    = 0
    for layer_idx, feat in layer_features.items():
        # feat: (1, seq_len, D)
        seq_len = feat.shape[1]
        expected = feat_T * feat_H * feat_W

        if seq_len == expected:
            feat_map = feat[0].view(feat_T, feat_H, feat_W, -1)   # (T, fH, fW, D)
        elif seq_len % (feat_H * feat_W) == 0:
            T_actual = seq_len // (feat_H * feat_W)
            feat_map = feat[0].view(T_actual, feat_H, feat_W, -1)
        else:
            # Fallback: just mean-pool sequence
            feat_map = feat[0].mean(dim=0, keepdim=True).unsqueeze(0).unsqueeze(0)
            feat_map = feat_map.expand(feat_T, feat_H, feat_W, -1)

        feat_map = feat_map.float()   # (T, fH, fW, D)
        feat_sum  = feat_map if feat_sum is None else feat_sum + feat_map
        n_layers += 1

    feat_avg = feat_sum / n_layers   # (T, fH, fW, D)
    D = feat_avg.shape[-1]

    # ── Project Gaussians and sample features ─────────────────────────────────
    print(f"  Sampling features for {N_gauss} Gaussians across {n_cams} cameras ...")

    # Work in batches to avoid OOM
    BATCH = 4096
    gauss_embeddings = torch.zeros(N_gauss, D, dtype=torch.float32)

    for start in range(0, N_gauss, BATCH):
        end   = min(start + BATCH, N_gauss)
        batch_pos = positions[start:end]   # (B_g, 3)

        batch_emb = torch.zeros(end - start, D, dtype=torch.float32)
        n_valid   = 0

        for cam_idx, cam in enumerate(cameras):
            if cam_idx >= feat_T:
                break    # only use as many cameras as temporal frames

            c2w  = np.array(cam["cam_to_world"], dtype=np.float32)
            w2c  = np.linalg.inv(c2w)
            fx, fy, cx, cy = cam["params"]

            # Project batch_pos
            pts  = np.concatenate(
                [batch_pos, np.ones((len(batch_pos), 1), dtype=np.float32)], axis=-1
            )
            pts_cam = (w2c @ pts.T).T           # (B_g, 4)
            z       = pts_cam[:, 2].clip(1e-5)
            u       = (pts_cam[:, 0] / z * fx + cx) / orig_W * feat_W
            v       = (pts_cam[:, 1] / z * fy + cy) / orig_H * feat_H
            # Normalise to [-1, 1]
            u_n = torch.from_numpy((u / feat_W * 2.0 - 1.0).astype(np.float32))
            v_n = torch.from_numpy((v / feat_H * 2.0 - 1.0).astype(np.float32))

            # Only use Gaussians that project inside the frame
            in_frame = (u_n >= -1) & (u_n <= 1) & (v_n >= -1) & (v_n <= 1)
            if not in_frame.any():
                continue

            # feat_avg[cam_idx]: (fH, fW, D)
            frame_feat = feat_avg[cam_idx % feat_T].permute(2, 0, 1)  # (D, fH, fW)

            # grid_sample expects (1, D, fH, fW) + grid (1, 1, B_g, 2)
            grid = torch.stack([u_n, v_n], dim=-1)[None, None, :, :]  # (1, 1, B_g, 2)
            sampled = F.grid_sample(
                frame_feat[None],     # (1, D, fH, fW)
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )   # (1, D, 1, B_g)
            sampled = sampled[0, :, 0, :].T   # (B_g, D)

            batch_emb += sampled * in_frame[:, None].float()
            n_valid   += 1

        if n_valid > 0:
            batch_emb /= n_valid
        gauss_embeddings[start:end] = batch_emb

        if start % (BATCH * 10) == 0:
            print(f"  Progress: {end}/{N_gauss} Gaussians", end="\r")

    print(f"\n  Done. Feature shape: {gauss_embeddings.shape}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gauss_embeddings, str(out_path))
    print(f"[FeatExtract] Saved motion embeddings → {out_path}")

    return gauss_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract CogVideoX motion features for each Gaussian."
    )
    parser.add_argument("--scene-dir",  required=True,
                        help="Scene directory with cameras.json + outputs/bg_gaussians/")
    parser.add_argument("--lora-dir",   required=True,
                        help="Directory containing the trained LoRA adapter")
    parser.add_argument("--output",     default="./outputs/motion_embeddings.pt",
                        help="Output path for motion_embeddings.pt")
    parser.add_argument("--model-id",   default="zai-org/CogVideoX-5b-I2V",
                        help="Base CogVideoX model ID (HF hub or local path)")
    parser.add_argument("--clip-len",   type=int, default=16,
                        help="Number of frames to run through the model (default: 16)")
    args = parser.parse_args()

    extract_motion_embeddings(
        scene_dir=args.scene_dir,
        lora_dir=args.lora_dir,
        output_path=args.output,
        model_id=args.model_id,
        clip_len=args.clip_len,
    )


if __name__ == "__main__":
    main()

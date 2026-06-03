"""
training/warm_init.py — Warm initialisation for Deformable-3DGS deformation MLP
=================================================================================
Trains a small linear projection (D_feat → 64) that maps CogVideoX motion
embeddings to the first-layer weights of the deformation MLP, supervised on
synthetic Blender scenes where ground-truth deformations are known.

Then applies the projection to the real scene's motion embeddings to produce
warm_deform_weights.pt — used as the MLP's initial weights instead of random.

Pipeline:
  1. Load motion_embeddings.pt + Blender GT deformations (from manifest)
  2. Train linear projection: emb → deform_init  (supervised, MSE loss)
  3. Apply projection to real scene embeddings
  4. Save warm_deform_weights.pt

FLAG: If training loss doesn't go below 0.05 after 50 epochs, the domain gap
between synthetic and real data may be too large. Come back and we will adjust
the projection architecture, add domain adaptation, or fall back to random init.

Usage:
    python training/warm_init.py \\
        --embeddings    ./outputs/motion_embeddings.pt \\
        --blender-dir   ./data/synthetic/ \\
        --output        ./outputs/warm_deform_weights.pt \\
        [--epochs 50]   [--lr 1e-3]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


# ─────────────────────────────────────────────────────────────────────────────
# Deformation MLP architecture (mirrors Deformable-3D-Gaussians)
# ─────────────────────────────────────────────────────────────────────────────

class DeformationMLP(nn.Module):
    """
    Replicated architecture of the Deformable-3DGS deformation network.
    Input:  (x, y, z, t) — 4D
    Output: (Δx, Δy, Δz, Δqw, Δqx, Δqy, Δqz, Δs0, Δs1, Δs2) — 10D

    VERIFY: Exact architecture from Deformable-3D-Gaussians/scene/deformation.py.
    The depth (8 layers), skip connections, and activation functions below match
    the typical NeRF-style MLP used in the repo.
    """

    def __init__(self, hidden: int = 256, skips: tuple = (4,)):
        super().__init__()
        self.skips = skips
        self.input_dim  = 4   # x, y, z, t
        self.output_dim = 10  # Δxyz + Δrot(4) + Δscale(3)

        layers = []
        in_dim = self.input_dim
        for i in range(8):
            if i in skips:
                in_dim += self.input_dim
            layers.append(nn.Linear(in_dim, hidden))
            in_dim = hidden
        self.layers = nn.ModuleList(layers)
        self.head   = nn.Linear(hidden, self.output_dim)
        self.act    = nn.ReLU(inplace=True)

    def forward(self, xyz: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Args:
            xyz: (N, 3) positions
            t:   (N, 1) normalised time
        Returns:
            delta_xyz:   (N, 3)
            delta_rot:   (N, 4)
            delta_scale: (N, 3)
        """
        x = torch.cat([xyz, t], dim=-1)   # (N, 4)
        inp = x
        h   = x
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([h, inp], dim=-1)
            h = self.act(layer(h))
        out = self.head(h)   # (N, 10)
        return out[:, :3], out[:, 3:7], out[:, 7:]


# ─────────────────────────────────────────────────────────────────────────────
# Linear projection: motion embeddings → MLP init weights
# ─────────────────────────────────────────────────────────────────────────────

class MotionProjection(nn.Module):
    """
    Maps a (D_feat,) motion embedding to a (256, 4) weight matrix
    (the first layer of the deformation MLP).

    Architecture:
        D_feat → 256 → 256 → 256 × 4   (flattened first-layer weights)
    """

    def __init__(self, d_feat: int, hidden_dim: int = 256):
        super().__init__()
        out_dim = hidden_dim * 4   # (hidden_dim neurons × 4 input dims)
        self.net = nn.Sequential(
            nn.Linear(d_feat, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb: (N, D_feat) per-Gaussian embeddings
        Returns:
            weights: (N, hidden_dim, 4) — first-layer weight matrix per Gaussian
        """
        out = self.net(emb)   # (N, hidden_dim * 4)
        return out.view(-1, self.hidden_dim, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Blender GT deformation dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_blender_gt_deformations(manifest_path: str, n_points: int = 2000,
                                  n_scenes: int = 100) -> tuple:
    """
    Extract ground-truth (position, time, deformation, scene_index) triples from
    Blender depth maps.

    Returns:
        pts:         (M, 3) float32 — world-space positions at t=0
        times:       (M, 1) float32 — normalised time [0, 1]
        deforms:     (M, 10) float32 — target deformation (Δxyz + zeros for rot/scale)
        scene_idxs:  (M,) int64 — which scene each point belongs to
    """
    import imageio
    from PIL import Image

    manifest = json.loads(Path(manifest_path).read_text())
    valid    = [s for s in manifest if "error" not in s][:n_scenes]

    all_pts     = []
    all_times   = []
    all_deforms = []
    all_sidxs   = []

    for sc_num, sc in enumerate(valid):
        depth_dir  = Path(sc["depth_dir"])
        render_dir = Path(sc["render_dir"])
        n_frames   = sc["n_frames"]

        depth_files = sorted(depth_dir.glob("*.exr"))
        if len(depth_files) < 2:
            continue

        # Use simple pinhole camera (approximate)
        W, H  = sc["resolution"]
        focal = max(W, H) * 1.2
        cx, cy = W / 2, H / 2

        def depth_to_world(depth_np, frame_idx, total_frames):
            """Unproject depth to 3D, assuming identity camera at t=0."""
            rows, cols = np.where(np.isfinite(depth_np) & (depth_np > 0))
            if len(rows) == 0:
                return None, None
            idx = np.random.choice(len(rows), min(n_points // n_frames, len(rows)),
                                   replace=False)
            rows, cols = rows[idx], cols[idx]
            z = depth_np[rows, cols]
            x = (cols - cx) / focal * z
            y = (rows - cy) / focal * z
            pts3d = np.stack([x, y, z], axis=-1).astype(np.float32)
            t = np.full((len(pts3d), 1), frame_idx / total_frames, dtype=np.float32)
            return pts3d, t

        def load_depth_exr(path: Path) -> np.ndarray:
            try:
                import imageio
                d = imageio.imread(str(path), format="EXR-FI")
                return d[:, :, 0] if d.ndim == 3 else d
            except Exception:
                try:
                    import cv2
                    d = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
                    return d.astype(np.float32) if d is not None else np.zeros((H, W))
                except Exception:
                    return np.zeros((H, W), dtype=np.float32)

        # Frame 0 as reference positions
        d0 = load_depth_exr(depth_files[0])
        pts0, _ = depth_to_world(d0, 0, n_frames)
        if pts0 is None:
            continue

        # Sample a few later frames and compute displacement
        sample_frames = np.linspace(1, len(depth_files) - 1, min(4, len(depth_files) - 1),
                                     dtype=int)
        for fidx in sample_frames:
            if fidx >= len(depth_files):
                continue
            d_t = load_depth_exr(depth_files[fidx])
            pts_t, t_t = depth_to_world(d_t, fidx, n_frames)
            if pts_t is None:
                continue

            # Match points by nearest neighbour (approximate GT displacement)
            n_match = min(len(pts0), len(pts_t), 512)
            idx0 = np.random.choice(len(pts0), n_match, replace=False)
            idx_t = np.random.choice(len(pts_t), n_match, replace=False)

            delta_xyz   = (pts_t[idx_t] - pts0[idx0]).astype(np.float32)
            deform_10   = np.concatenate(
                [delta_xyz, np.zeros((n_match, 7), dtype=np.float32)],
                axis=-1
            )

            all_pts.append(pts0[idx0])
            all_times.append(t_t[idx_t])
            all_deforms.append(deform_10)
            all_sidxs.append(np.full(n_match, sc_num, dtype=np.int64))

    if not all_pts:
        raise RuntimeError(
            "No valid Blender ground-truth deformations found. "
            "Check that blender_render.py produced depth EXR files."
        )

    pts      = np.concatenate(all_pts,     axis=0)
    times    = np.concatenate(all_times,   axis=0)
    deforms  = np.concatenate(all_deforms, axis=0)
    sidxs    = np.concatenate(all_sidxs,   axis=0)
    print(f"  Loaded {len(pts)} (position, time, deformation) triples from {len(valid)} scenes")
    return pts, times, deforms, sidxs


# ─────────────────────────────────────────────────────────────────────────────
# Per-scene CogVideoX embedding extraction (synthetic scenes)
# ─────────────────────────────────────────────────────────────────────────────

def extract_synthetic_scene_embeddings(
    manifest_path: str,
    model_id: str = "zai-org/CogVideoX-5b-I2V",
    lora_dir: str | None = None,
    n_scenes: int = 100,
    clip_len: int = 8,
    size: int = 256,
    target_blocks: tuple = (12, 13, 14, 15, 16, 17, 18),
    cache_path: str | None = None,
) -> torch.Tensor:
    """
    Extract a single global CogVideoX embedding (D,) per Blender scene.

    Returns:
        embeddings: (n_valid_scenes, D) float32 — one embedding per scene,
                    in the same order as valid scenes in the manifest.
    """
    import torchvision.transforms.functional as TF
    from PIL import Image
    from diffusers import CogVideoXPipeline

    # Check cache
    if cache_path and Path(cache_path).exists():
        print(f"[WarmInit] Loading cached synthetic embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[WarmInit] Loading CogVideoX for synthetic embedding extraction ...")
    pipeline = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    vae = pipeline.vae.to(device).eval()
    transformer = pipeline.transformer.to(device).eval()
    hidden_size = getattr(transformer.config, "hidden_size", 3072)

    if lora_dir:
        from peft import PeftModel
        lora_path = Path(lora_dir) / "lora_adapter"
        if lora_path.exists():
            transformer = PeftModel.from_pretrained(transformer, str(lora_path))
            print(f"  Loaded LoRA from {lora_path}")

    for p in transformer.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)

    manifest = json.loads(Path(manifest_path).read_text())
    valid = [s for s in manifest if "error" not in s][:n_scenes]

    scene_embeddings = []

    for sc_num, sc in enumerate(valid):
        render_dir = Path(sc["render_dir"])
        frame_files = sorted(render_dir.glob("frame_*.png"))[:clip_len]
        if not frame_files:
            scene_embeddings.append(torch.zeros(hidden_size))
            continue

        frames = []
        for fp in frame_files:
            img = Image.open(fp).convert("RGB").resize((size, size), Image.BILINEAR)
            frames.append(TF.to_tensor(img) * 2.0 - 1.0)
        while len(frames) < clip_len:
            frames.append(frames[-1].clone())

        video = torch.stack(frames).to(device, dtype=torch.bfloat16)   # (T, 3, H, W)

        with torch.no_grad():
            latents = vae.encode(video).latent_dist.sample()
            latents = latents * vae.config.scaling_factor              # (T, C, h, w)
            latent = latents.unsqueeze(0).permute(0, 2, 1, 3, 4)      # (1, C, T, h, w)

            layer_feats: dict[int, torch.Tensor] = {}

            def make_hook(i: int):
                def hook(mod, inp, out):
                    feat = out[0] if isinstance(out, tuple) else out
                    layer_feats[i] = feat.detach().mean(dim=(0, 1)).cpu()  # (D,)
                return hook

            hooks = [
                transformer.transformer_blocks[i].register_forward_hook(make_hook(i))
                for i in target_blocks
                if i < len(transformer.transformer_blocks)
            ]
            try:
                _ = transformer(
                    hidden_states=latent,
                    timestep=torch.zeros(1, dtype=torch.long, device=device),
                    encoder_hidden_states=torch.zeros(
                        1, 1, hidden_size, device=device, dtype=torch.bfloat16
                    ),
                    return_dict=False,
                )
            finally:
                for h in hooks:
                    h.remove()

        if layer_feats:
            scene_emb = torch.stack(list(layer_feats.values())).mean(dim=0)  # (D,)
        else:
            scene_emb = torch.zeros(hidden_size)

        scene_embeddings.append(scene_emb.float())
        print(f"  Scene {sc_num+1}/{len(valid)}", end="\r")

    print()
    result = torch.stack(scene_embeddings, dim=0)   # (n_scenes, D)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_path)
        print(f"  Cached synthetic embeddings → {cache_path}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main training + application
# ─────────────────────────────────────────────────────────────────────────────

def train_projection(
    embeddings_path: str,
    blender_manifest: str,
    output_path: str,
    epochs: int    = 50,
    lr: float      = 1e-3,
    batch_size: int = 256,
    synthetic_embeddings_path: str | None = None,
    model_id: str = "zai-org/CogVideoX-5b-I2V",
    lora_dir: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load real-scene motion embeddings (for final weight generation) ───────
    print(f"[WarmInit] Loading real-scene embeddings from {embeddings_path} ...")
    real_embeddings = torch.load(embeddings_path, map_location="cpu")  # (N, D)
    N_gauss, D_feat = real_embeddings.shape
    print(f"  Shape: {N_gauss} Gaussians × {D_feat}-dim features")

    # ── Load synthetic GT deformations ────────────────────────────────────────
    print(f"[WarmInit] Loading Blender GT deformations from {blender_manifest} ...")
    pts_np, times_np, deforms_np, sidxs_np = load_blender_gt_deformations(blender_manifest)

    n_syn_scenes = int(sidxs_np.max()) + 1

    pts     = torch.from_numpy(pts_np).to(device)
    times   = torch.from_numpy(times_np).to(device)
    deforms = torch.from_numpy(deforms_np).to(device)
    sidxs   = torch.from_numpy(sidxs_np)   # kept on CPU for indexing

    # ── Get or compute synthetic scene embeddings ─────────────────────────────
    syn_emb_cache = str(Path(blender_manifest).parent / "synthetic_scene_embeddings.pt")
    if synthetic_embeddings_path and Path(synthetic_embeddings_path).exists():
        syn_scene_embs = torch.load(synthetic_embeddings_path, map_location="cpu")
        print(f"  Loaded synthetic embeddings from {synthetic_embeddings_path}")
    else:
        print("[WarmInit] Extracting CogVideoX embeddings from Blender scenes ...")
        print("  (This runs once; result cached at synthetic_scene_embeddings.pt)")
        syn_scene_embs = extract_synthetic_scene_embeddings(
            manifest_path=blender_manifest,
            model_id=model_id,
            lora_dir=lora_dir,
            n_scenes=n_syn_scenes,
            cache_path=syn_emb_cache,
        )

    # syn_scene_embs: (n_scenes, D)
    if syn_scene_embs.shape[0] < n_syn_scenes:
        # Pad with zeros if fewer scenes were extracted
        pad = torch.zeros(n_syn_scenes - syn_scene_embs.shape[0], D_feat)
        syn_scene_embs = torch.cat([syn_scene_embs, pad], dim=0)

    syn_scene_embs = syn_scene_embs.to(device)

    # ── Deformation MLP ───────────────────────────────────────────────────────
    mlp = DeformationMLP(hidden=256).to(device)

    # ── Projection network ────────────────────────────────────────────────────
    proj = MotionProjection(d_feat=D_feat, hidden_dim=256).to(device)
    opt  = AdamW(proj.parameters(), lr=lr)

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"[WarmInit] Training projection for {epochs} epochs ...")

    n_total   = len(pts)
    best_loss = float("inf")

    for epoch in range(epochs):
        perm = torch.randperm(n_total)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_total, batch_size):
            idx    = perm[start : start + batch_size]
            b_pts  = pts[idx]
            b_time = times[idx]
            b_def  = deforms[idx]

            # Use embeddings from the CORRECT synthetic scene for each point.
            # sidxs[idx] gives which Blender scene each point came from.
            b_scene_idx = sidxs[idx]   # (batch,) int64
            b_emb = syn_scene_embs[b_scene_idx]   # (batch, D) — proper correspondence

            W_pred = proj(b_emb)   # (batch, 256, 4)

            inp = torch.cat([b_pts, b_time], dim=-1)   # (batch, 4)
            h0  = F.relu(torch.bmm(W_pred, inp.unsqueeze(-1)).squeeze(-1))  # (batch, 256)

            with torch.no_grad():
                h = h0
                for i, layer in enumerate(mlp.layers[1:], start=1):
                    if i in mlp.skips:
                        h = torch.cat([h, inp], dim=-1)
                    h = F.relu(layer(h))
                out = mlp.head(h)   # (batch, 10)

            loss = F.mse_loss(out, b_def)
            loss.backward()
            opt.step()
            opt.zero_grad()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # ── FLAG ──────────────────────────────────────────────────────────────────
    FLAG_THRESHOLD = 0.05
    if best_loss > FLAG_THRESHOLD:
        print(
            f"\n  FLAG: Best training loss = {best_loss:.4f} > {FLAG_THRESHOLD}.\n"
            f"  Domain gap between synthetic and real data may be too large.\n"
            f"  Options: (a) add domain adaptation, (b) lower lr, "
            f"(c) deeper projection, (d) fall back to random init.\n"
        )
    else:
        print(f"\n  Good convergence: best loss = {best_loss:.4f} < {FLAG_THRESHOLD}")

    # ── Apply projection to real scene embeddings ─────────────────────────────
    print(f"\n[WarmInit] Generating warm weights for {N_gauss} real-scene Gaussians ...")
    proj.eval()
    with torch.no_grad():
        emb_batches = real_embeddings.to(device).split(1024)
        W_all = torch.cat([proj(b) for b in emb_batches], dim=0)   # (N, 256, 4)
        W_mean = W_all.mean(dim=0)   # (256, 4)

    warm_state = mlp.state_dict()
    warm_state["layers.0.weight"] = W_mean.cpu()
    warm_state["layers.0.bias"]   = torch.zeros(256)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(warm_state, str(out_path))
    print(f"[WarmInit] Saved warm weights → {out_path}")

    return best_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train warm initialisation for Deformable-3DGS deformation MLP."
    )
    parser.add_argument("--embeddings",   required=True,
                        help="Path to real-scene motion_embeddings.pt")
    parser.add_argument("--blender-dir",  default=None,
                        help="Path to Blender manifest.json OR its parent directory")
    parser.add_argument("--output",       default="./outputs/warm_deform_weights.pt",
                        help="Output path for warm_deform_weights.pt")
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch-size",   type=int, default=256)
    parser.add_argument("--synthetic-embeddings", default=None,
                        help="Pre-computed synthetic scene embeddings .pt (optional; "
                             "extracted automatically if missing)")
    parser.add_argument("--model-id",     default="zai-org/CogVideoX-5b-I2V",
                        help="CogVideoX model ID for synthetic embedding extraction")
    parser.add_argument("--lora-dir",     default=None,
                        help="LoRA adapter directory (optional, for extraction)")
    args = parser.parse_args()

    if args.blender_dir is None:
        parser.error("--blender-dir is required")
    bd = Path(args.blender_dir)
    manifest = bd if bd.name == "manifest.json" else bd / "manifest.json"
    if not manifest.exists():
        parser.error(f"manifest.json not found: {manifest}")

    train_projection(
        embeddings_path=args.embeddings,
        blender_manifest=str(manifest),
        output_path=args.output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        synthetic_embeddings_path=args.synthetic_embeddings,
        model_id=args.model_id,
        lora_dir=args.lora_dir,
    )


if __name__ == "__main__":
    main()

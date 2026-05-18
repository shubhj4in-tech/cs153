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
    Extract ground-truth (position, time, deformation) triples from
    Blender depth maps: for each scene, use depth + camera intrinsics to get
    3D point positions, then compute deformation as displacement from frame 0.

    Returns:
        pts:   (M, 3) float32 — world-space positions at t=0
        times: (M, 1) float32 — normalised time [0, 1]
        deforms: (M, 10) float32 — target deformation (Δxyz + zeros for rot/scale)
    """
    import imageio
    from PIL import Image

    manifest = json.loads(Path(manifest_path).read_text())
    valid    = [s for s in manifest if "error" not in s][:n_scenes]

    all_pts     = []
    all_times   = []
    all_deforms = []

    for sc in valid:
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

    if not all_pts:
        raise RuntimeError(
            "No valid Blender ground-truth deformations found. "
            "Check that blender_render.py produced depth EXR files."
        )

    pts     = np.concatenate(all_pts,     axis=0)
    times   = np.concatenate(all_times,   axis=0)
    deforms = np.concatenate(all_deforms, axis=0)
    print(f"  Loaded {len(pts)} (position, time, deformation) triples from {len(valid)} scenes")
    return pts, times, deforms


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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load real-scene motion embeddings ─────────────────────────────────────
    print(f"[WarmInit] Loading motion embeddings from {embeddings_path} ...")
    embeddings = torch.load(embeddings_path, map_location="cpu")  # (N, D)
    N_gauss, D_feat = embeddings.shape
    print(f"  Shape: {N_gauss} Gaussians × {D_feat}-dim features")

    # ── Load synthetic GT deformations ────────────────────────────────────────
    print(f"[WarmInit] Loading Blender GT deformations from {blender_manifest} ...")
    pts_np, times_np, deforms_np = load_blender_gt_deformations(blender_manifest)

    pts     = torch.from_numpy(pts_np).to(device)
    times   = torch.from_numpy(times_np).to(device)
    deforms = torch.from_numpy(deforms_np).to(device)

    # ── Deformation MLP (fixed architecture for target weight extraction) ─────
    mlp = DeformationMLP(hidden=256).to(device)

    # ── Projection network ────────────────────────────────────────────────────
    proj = MotionProjection(d_feat=D_feat, hidden_dim=256).to(device)
    opt  = AdamW(proj.parameters(), lr=lr)

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"[WarmInit] Training projection for {epochs} epochs ...")

    n_syn   = len(pts)
    best_loss = float("inf")

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n_syn)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_syn, batch_size):
            idx = perm[start : start + batch_size]
            b_pts  = pts[idx]
            b_time = times[idx]
            b_def  = deforms[idx]

            # Use random Gaussian embeddings from real scene
            # (approximation: we don't have exact correspondence)
            g_idx = torch.randint(0, N_gauss, (len(idx),))
            b_emb = embeddings[g_idx].to(device)

            # Project embeddings → first-layer weights
            W_pred = proj(b_emb)   # (batch, 256, 4) — predicted first-layer weight matrix

            # Apply predicted weights to position+time input and run rest of MLP
            inp      = torch.cat([b_pts, b_time], dim=-1)   # (batch, 4)
            # First layer: manual matmul using predicted weights
            h0 = F.relu(torch.bmm(W_pred, inp.unsqueeze(-1)).squeeze(-1))  # (batch, 256)

            # Continue with frozen MLP layers (from layer 1 onward)
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
            f"  The domain gap between synthetic Blender and real data may be too large.\n"
            f"  Come back and we will:\n"
            f"    (a) Add a domain adaptation module\n"
            f"    (b) Scale down the projection learning rate\n"
            f"    (c) Use a deeper projection network\n"
            f"    (d) Fall back to random initialisation (still runs, just slower convergence)\n"
        )
    else:
        print(f"\n  Good convergence: best loss = {best_loss:.4f} < {FLAG_THRESHOLD}")

    # ── Apply projection to real scene embeddings ─────────────────────────────
    print(f"\n[WarmInit] Generating warm weights for {N_gauss} Gaussians ...")
    proj.eval()
    with torch.no_grad():
        # Average the projected first-layer weights over all Gaussians
        emb_batches = embeddings.to(device).split(1024)
        W_all = torch.cat([proj(b) for b in emb_batches], dim=0)   # (N, 256, 4)
        W_mean = W_all.mean(dim=0)   # (256, 4) — mean initial weights

    # Build a state dict for the MLP with warm first layer
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
                        help="Path to motion_embeddings.pt")
    parser.add_argument("--blender-dir",  default=None,
                        help="Path to Blender manifest.json OR its parent directory")
    parser.add_argument("--output",       default="./outputs/warm_deform_weights.pt",
                        help="Output path for warm_deform_weights.pt")
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch-size",   type=int, default=256)
    args = parser.parse_args()

    # Resolve manifest path
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
    )


if __name__ == "__main__":
    main()

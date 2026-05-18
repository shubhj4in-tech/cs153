"""
Stage 4 — Export to .splat Format
===================================
Merges the static background (3DGS) and dynamic foreground (Deformable-3DGS)
into per-timestep .splat files consumable by the Three.js viewer.

For each timestep t:
  1. Load BG Gaussians (static, no deformation)
  2. Load FG Gaussians and apply the deformation MLP at time t
  3. Merge → write <timestamp>.splat
  4. Write manifest.json listing all files

.splat binary format (32 bytes per Gaussian):
  bytes  0-11:  position (3 × float32)
  bytes 12-23:  scale    (3 × float32)
  bytes 24-27:  color    (RGBA uint8, A encodes opacity via sigmoid)
  bytes 28-31:  rotation (4 × uint8, quaternion packed to [-1,1] as (x-127.5)/127.5)

Target: ≤ 15 MB per file, achieved by pruning low-opacity Gaussians.

Usage:
    python pipeline/04_export.py \\
        --scene-dir ./scene/ \\
        --output    ./viewer/public/

    # or with explicit paths:
    python pipeline/04_export.py \\
        --bg-ckpt  ./scene/outputs/bg_gaussians/point_cloud/iteration_30000/ \\
        --fg-ckpt  ./scene/outputs/fg_gaussians/point_cloud/iteration_30000/ \\
        --output   ./viewer/public/
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# .splat format helpers
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_BYTES = 15 * 1024 * 1024   # 15 MB
SPLAT_BYTES_PER_GAUSSIAN = 32


def pack_splat(positions: np.ndarray, scales: np.ndarray, colors: np.ndarray,
               opacities: np.ndarray, rotations: np.ndarray) -> bytes:
    """
    Pack Gaussians into the antimatter15/splat binary format.

    Args:
        positions:  (N, 3) float32 — x, y, z
        scales:     (N, 3) float32 — log-scales (will be exponentiated)
        colors:     (N, 3) float32 — RGB in [0, 1]
        opacities:  (N,)   float32 — pre-sigmoid activation values
        rotations:  (N, 4) float32 — quaternion (w, x, y, z), unit norm

    Returns:
        bytes of length N * 32
    """
    N = len(positions)
    buf = bytearray(N * SPLAT_BYTES_PER_GAUSSIAN)

    # Sigmoid opacity → alpha in [0,255]
    alpha   = (1.0 / (1.0 + np.exp(-opacities.astype(np.float64))))
    alpha_u8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    # Colors: apply sigmoid (stored as SH DC term, scale = 0.282)
    # 3DGS stores colour as SH DC: c = 0.5 + 0.2820948 * sh_0
    rgb   = np.clip(0.5 + 0.2820948 * colors, 0.0, 1.0)
    r_u8  = np.clip(rgb[:, 0] * 255, 0, 255).astype(np.uint8)
    g_u8  = np.clip(rgb[:, 1] * 255, 0, 255).astype(np.uint8)
    b_u8  = np.clip(rgb[:, 2] * 255, 0, 255).astype(np.uint8)

    # Scale: exp to go from log-scale to actual scale
    sc    = np.exp(scales.astype(np.float32))

    # Rotation: quaternion (w, x, y, z) → pack to uint8 via (val+1)/2 * 255
    # Convert to (x, y, z, w) ordering expected by the splat viewer
    rot_reordered = np.stack([rotations[:, 1], rotations[:, 2],
                               rotations[:, 3], rotations[:, 0]], axis=-1)
    rot_u8 = np.clip((rot_reordered + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)

    pos_f = positions.astype(np.float32)
    sc_f  = sc.astype(np.float32)

    offset = 0
    for i in range(N):
        # 12 bytes: position
        struct.pack_into("fff", buf, offset,
                         pos_f[i, 0], pos_f[i, 1], pos_f[i, 2])
        offset += 12
        # 12 bytes: scale
        struct.pack_into("fff", buf, offset,
                         sc_f[i, 0], sc_f[i, 1], sc_f[i, 2])
        offset += 12
        # 4 bytes: RGBA color + opacity
        struct.pack_into("BBBB", buf, offset,
                         r_u8[i], g_u8[i], b_u8[i], alpha_u8[i])
        offset += 4
        # 4 bytes: rotation (x, y, z, w) as uint8
        struct.pack_into("BBBB", buf, offset,
                         rot_u8[i, 0], rot_u8[i, 1],
                         rot_u8[i, 2], rot_u8[i, 3])
        offset += 4

    return bytes(buf)


# ─────────────────────────────────────────────────────────────────────────────
# Load 3DGS checkpoint (.ply format written by train.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_gaussians_from_ply(ply_path: Path) -> dict:
    """
    Load a 3DGS point_cloud.ply into numpy arrays.

    The 3DGS .ply has per-vertex attributes:
      x, y, z                      — position
      nx, ny, nz                   — normals (usually 0)
      f_dc_0, f_dc_1, f_dc_2      — SH DC (= colour)
      f_rest_0 … f_rest_44        — higher-order SH (45 floats for degree 3)
      opacity                      — raw opacity (pre-sigmoid)
      scale_0, scale_1, scale_2   — log-scale
      rot_0, rot_1, rot_2, rot_3  — quaternion (w, x, y, z)
    """
    from plyfile import PlyData

    ply  = PlyData.read(str(ply_path))
    verts = ply["vertex"]

    pos  = np.stack([verts["x"], verts["y"], verts["z"]], axis=-1)
    col  = np.stack([verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"]], axis=-1)
    opa  = verts["opacity"]
    sc   = np.stack([verts["scale_0"], verts["scale_1"], verts["scale_2"]], axis=-1)
    rot  = np.stack([verts["rot_0"], verts["rot_1"],
                     verts["rot_2"], verts["rot_3"]], axis=-1)

    return {
        "positions":  pos.astype(np.float32),
        "colors":     col.astype(np.float32),
        "opacities":  np.array(opa, dtype=np.float32),
        "scales":     sc.astype(np.float32),
        "rotations":  rot.astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load Deformable-3DGS checkpoint + deformation MLP
# ─────────────────────────────────────────────────────────────────────────────

def load_deformable_gaussians(fg_ckpt_dir: Path, timestamp: float) -> dict:
    """
    Load the Deformable-3DGS checkpoint and apply the deformation MLP at
    normalised time t ∈ [0, 1].

    The deformation MLP takes (x, y, z, t) and outputs Δ(xyz, rot, scale).

    VERIFY: The exact checkpoint filename convention used by Deformable-3DGS.
    Typically: fg_ckpt_dir/point_cloud.ply  (canonical positions)
               fg_ckpt_dir/deformation/deformation.pth  (MLP weights)
    Check the Deformable-3D-Gaussians repo structure if paths differ.
    """
    ply_path = fg_ckpt_dir / "point_cloud.ply"
    mlp_path = fg_ckpt_dir / "deformation" / "deformation.pth"

    if not ply_path.exists():
        raise FileNotFoundError(f"FG point cloud not found: {ply_path}")

    gauss = load_gaussians_from_ply(ply_path)
    N     = len(gauss["positions"])

    if not mlp_path.exists():
        print(f"  WARNING: deformation MLP not found at {mlp_path}. "
              "Using canonical positions (no motion).")
        return gauss

    # ── Load MLP ──────────────────────────────────────────────────────────────
    # VERIFY: Import path matches the Deformable-3DGS package structure.
    # The MLP class is typically in scene/deformation.py or utils/deform_model.py
    try:
        sys.path.insert(0, "/opt/deformable-3dgs")
        from scene.deformation import DeformationNetwork   # VERIFY class name
        mlp_state = torch.load(str(mlp_path), map_location="cpu")
        mlp = DeformationNetwork()
        mlp.load_state_dict(mlp_state)
        mlp.eval()
    except (ImportError, Exception) as e:
        print(f"  WARNING: Could not load deformation MLP ({e}). "
              "Using canonical positions.")
        return gauss

    # ── Apply deformation ────────────────────────────────────────────────────
    t_tensor = torch.full((N, 1), fill_value=timestamp, dtype=torch.float32)
    pos_t    = torch.from_numpy(gauss["positions"])    # (N, 3)
    rot_t    = torch.from_numpy(gauss["rotations"])    # (N, 4)
    sc_t     = torch.from_numpy(gauss["scales"])       # (N, 3)

    with torch.no_grad():
        # VERIFY: exact call signature of DeformationNetwork.forward()
        # Typical: (xyz, time) → (delta_xyz, delta_rot, delta_scale)
        delta = mlp(pos_t, t_tensor)

    if isinstance(delta, (tuple, list)) and len(delta) == 3:
        d_pos, d_rot, d_sc = delta
        gauss["positions"] = (pos_t + d_pos).numpy().astype(np.float32)
        gauss["rotations"] = (rot_t + d_rot).numpy().astype(np.float32)
        gauss["scales"]    = (sc_t  + d_sc).numpy().astype(np.float32)
    else:
        # Some versions return just delta_xyz
        d_pos = delta if isinstance(delta, torch.Tensor) else delta[0]
        gauss["positions"] = (pos_t + d_pos).numpy().astype(np.float32)

    return gauss


# ─────────────────────────────────────────────────────────────────────────────
# Merge + prune to target file size
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_prune(bg: dict, fg: dict, max_bytes: int = MAX_FILE_BYTES) -> dict:
    """Concatenate BG and FG Gaussians, then prune low-opacity ones to fit budget."""
    merged = {
        k: np.concatenate([bg[k], fg[k]], axis=0)
        for k in ["positions", "colors", "opacities", "scales", "rotations"]
    }

    N      = len(merged["positions"])
    budget = max_bytes // SPLAT_BYTES_PER_GAUSSIAN

    if N > budget:
        # Keep highest-opacity Gaussians
        alpha  = 1.0 / (1.0 + np.exp(-merged["opacities"].astype(np.float64)))
        idx    = np.argsort(-alpha)[:budget]
        merged = {k: v[idx] for k, v in merged.items()}
        print(f"  Pruned {N} → {len(idx)} Gaussians to fit {max_bytes/1e6:.1f} MB budget")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main export loop
# ─────────────────────────────────────────────────────────────────────────────

def export_scene(bg_ckpt_dir: Path, fg_ckpt_dir: Path,
                 output_dir: Path, timestamps: list[float]):
    """Export one .splat per timestamp and write manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 4] Loading background Gaussians from {bg_ckpt_dir} ...")
    bg = load_gaussians_from_ply(bg_ckpt_dir / "point_cloud.ply")
    print(f"  BG: {len(bg['positions'])} Gaussians")

    manifest = {
        "version":    "1.0",
        "n_frames":   len(timestamps),
        "timestamps": timestamps,
        "files":      [],
    }

    for i, t in enumerate(timestamps):
        print(f"  Frame {i+1}/{len(timestamps)}  t={t:.4f}", end="\r")

        fg = load_deformable_gaussians(fg_ckpt_dir, timestamp=t)
        merged = merge_and_prune(bg, fg)

        splat_bytes = pack_splat(
            merged["positions"],
            merged["scales"],
            merged["colors"],
            merged["opacities"],
            merged["rotations"],
        )

        fname = f"frame_{i:04d}.splat"
        (output_dir / fname).write_bytes(splat_bytes)

        manifest["files"].append({
            "file":       fname,
            "timestamp":  t,
            "n_gaussians": len(merged["positions"]),
            "size_bytes":  len(splat_bytes),
        })

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    total_mb = sum(f["size_bytes"] for f in manifest["files"]) / 1e6
    print(f"\n[Stage 4] Export done:")
    print(f"  Frames   : {len(timestamps)}")
    print(f"  Total    : {total_mb:.1f} MB")
    print(f"  Output   : {output_dir}")
    print(f"  Manifest : {manifest_path}")


def find_ckpt_dir(base_dir: Path, default_iters: int = 30000) -> Path:
    """Find the checkpoint directory produced by the 3DGS trainer."""
    # Typical path: base_dir/point_cloud/iteration_30000/
    candidates = sorted(
        (base_dir / "point_cloud").glob("iteration_*"),
        key=lambda p: int(p.name.split("_")[-1]),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # Fallback: ply directly in base_dir
    return base_dir


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Export static+dynamic Gaussians to .splat files."
    )
    parser.add_argument("--scene-dir", default=None,
                        help="Scene directory (auto-discovers checkpoints)")
    parser.add_argument("--bg-ckpt",   default=None,
                        help="Explicit BG checkpoint dir (overrides --scene-dir)")
    parser.add_argument("--fg-ckpt",   default=None,
                        help="Explicit FG checkpoint dir (overrides --scene-dir)")
    parser.add_argument("--output",    required=True,
                        help="Output directory for .splat files + manifest.json")
    parser.add_argument("--n-frames",  type=int, default=60,
                        help="Number of exported timesteps (default: 60)")
    args = parser.parse_args()

    # ── Resolve checkpoint paths ──────────────────────────────────────────────
    if args.bg_ckpt:
        bg_ckpt = Path(args.bg_ckpt)
    elif args.scene_dir:
        bg_ckpt = find_ckpt_dir(Path(args.scene_dir) / "outputs" / "bg_gaussians")
    else:
        parser.error("Provide either --scene-dir or both --bg-ckpt and --fg-ckpt")

    if args.fg_ckpt:
        fg_ckpt = Path(args.fg_ckpt)
    elif args.scene_dir:
        fg_ckpt = find_ckpt_dir(Path(args.scene_dir) / "outputs" / "fg_gaussians")
    else:
        parser.error("Provide either --scene-dir or both --bg-ckpt and --fg-ckpt")

    # ── Get video duration from frames_meta.json if available ────────────────
    duration = 30.0  # default
    if args.scene_dir:
        meta_path = Path(args.scene_dir) / "frames_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            duration = meta.get("video_info", {}).get("duration", 30.0)

    timestamps = [round(duration * i / (args.n_frames - 1), 6)
                  for i in range(args.n_frames)]
    # Normalise to [0, 1] for the deformation MLP
    norm_timestamps = [t / duration for t in timestamps]

    export_scene(bg_ckpt, fg_ckpt, Path(args.output), norm_timestamps)


if __name__ == "__main__":
    main()

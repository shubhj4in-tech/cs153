"""
Stage 3 — 3D Reconstruction
============================
Two Modal A100 jobs:

  reconstruct_background()
      Trains 3DGS on bg_frames/ using the COLMAP-format cameras.json.
      ~30k iterations, saves to outputs/bg_gaussians/

  reconstruct_foreground()
      Trains Deformable-3DGS on fg_frames/ + fg_masks/ + cameras.json.
      Baseline: random MLP init, 30k iterations.
      Warm-init:  load warm_deform_weights.pt, 500 iterations.
      Saves to outputs/fg_gaussians/

Cost estimates:
  Background  : ~1.5 A100-hours ($2.25) at 30k iterations
  Foreground  : ~1.5 A100-hours ($2.25) baseline  |  ~0.025 A100-hours warm-init

Usage:
    # Background only
    python pipeline/03_reconstruct.py --mode bg --scene-dir ./scene/

    # Foreground baseline
    python pipeline/03_reconstruct.py --mode fg --scene-dir ./scene/

    # Foreground warm-init
    python pipeline/03_reconstruct.py --mode fg --scene-dir ./scene/ \\
        --warm-init ./outputs/warm_deform_weights.pt

    # Both sequentially
    python pipeline/03_reconstruct.py --mode both --scene-dir ./scene/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import modal
from modal_config import app, training_image, VOLUME_MAP, DATA_PATH, data_volume


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: write COLMAP-style data that the 3DGS train.py expects
# ─────────────────────────────────────────────────────────────────────────────

def prepare_colmap_structure(scene_dir: Path, frames_subdir: str,
                              output_colmap_dir: Path):
    """
    The official gaussian-splatting train.py expects one of:
      - A COLMAP database + sparse/0/{cameras,images,points3D}.{bin|txt}
      - OR a transforms.json (NeRF synthetic format)

    We produce the TEXT format from our cameras.json.
    cameras.json format:  list of {id, name, model, width, height, params, cam_to_world}
    """
    import numpy as np

    cameras_json = scene_dir / "cameras.json"
    if not cameras_json.exists():
        raise FileNotFoundError(f"cameras.json not found in {scene_dir}. Run Stage 1.")

    cams = json.loads(cameras_json.read_text())
    sparse_dir = output_colmap_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # ── cameras.txt ──────────────────────────────────────────────────────────
    # Format: camera_id  PINHOLE  width height fx fy cx cy
    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID  MODEL  WIDTH  HEIGHT  PARAMS[]\n")
        # All frames share the same intrinsics — use camera 1
        c0 = cams[0]
        fx, fy, cx, cy = c0["params"]
        f.write(f"1 PINHOLE {c0['width']} {c0['height']} "
                f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    # ── images.txt ───────────────────────────────────────────────────────────
    # Format: image_id  QW QX QY QZ TX TY TZ  camera_id  name
    # Rotation is world-to-camera; we have cam-to-world, so invert.
    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID  QW QX QY QZ TX TY TZ  CAMERA_ID  NAME\n")
        f.write("#   POINTS2D[]  as (X, Y, POINT3D_ID)\n")
        for cam in cams:
            c2w = np.array(cam["cam_to_world"])         # (4, 4)
            w2c = np.linalg.inv(c2w)
            R   = w2c[:3, :3]
            t   = w2c[:3, 3]
            # Convert rotation matrix to quaternion (w, x, y, z)
            q = rotation_matrix_to_quaternion(R)
            f.write(
                f"{cam['id']+1} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} "
                f"1 {cam['name']}\n"
            )
            f.write("\n")   # empty points line

    # ── points3D.txt ─────────────────────────────────────────────────────────
    ply_src = scene_dir / "pointcloud.ply"
    if ply_src.exists():
        import plyfile
        ply = plyfile.PlyData.read(str(ply_src))
        verts = ply["vertex"]
        with open(sparse_dir / "points3D.txt", "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID  X  Y  Z  R  G  B  ERROR  TRACK[]\n")
            for i, v in enumerate(verts):
                f.write(
                    f"{i+1} {float(v['x']):.6f} {float(v['y']):.6f} {float(v['z']):.6f} "
                    f"{int(v['red'])} {int(v['green'])} {int(v['blue'])} 1.0\n"
                )
    else:
        (sparse_dir / "points3D.txt").write_text(
            "# 3D point list — empty (no pointcloud.ply found)\n"
        )

    # Symlink / copy images
    images_dir = output_colmap_dir / "images"
    images_dir.mkdir(exist_ok=True)
    src_frames = scene_dir / frames_subdir
    for fp in sorted(src_frames.glob("*.png")):
        dst = images_dir / fp.name
        if not dst.exists():
            os.symlink(str(fp.resolve()), str(dst))


def rotation_matrix_to_quaternion(R):
    """Convert 3×3 rotation matrix to (w, x, y, z) quaternion."""
    import numpy as np
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / (tr + 1.0) ** 0.5
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    norm = (w**2 + x**2 + y**2 + z**2) ** 0.5
    return [w / norm, x / norm, y / norm, z / norm]


# ─────────────────────────────────────────────────────────────────────────────
# Background reconstruction (3DGS)
# ─────────────────────────────────────────────────────────────────────────────

def _do_reconstruct_background(scene_dir: Path):
    """Called inside the Modal container (or --local)."""
    import wandb

    colmap_dir = scene_dir / "_colmap_bg"
    prepare_colmap_structure(scene_dir, "bg_frames", colmap_dir)

    output_dir = scene_dir / "outputs" / "bg_gaussians"
    output_dir.mkdir(parents=True, exist_ok=True)

    gs_train = Path("/opt/gaussian-splatting/train.py")
    if not gs_train.exists():
        raise FileNotFoundError(
            "/opt/gaussian-splatting/train.py not found. "
            "Check that the Modal image built correctly."
        )

    cmd = [
        sys.executable, str(gs_train),
        "-s", str(colmap_dir),
        "-m", str(output_dir),
        "--iterations", "30000",
        "--densify_until_iter", "15000",
        "--test_iterations", "7000", "15000", "30000",
        "--save_iterations", "15000", "30000",
    ]

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "4drecon"),
            name=f"bg_gs_{scene_dir.name}",
            config={"iterations": 30000, "mode": "background"},
        )

    print(f"[Stage 3] Launching 3DGS training:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    print(f"[Stage 3] Background reconstruction done → {output_dir}")

    if wandb_key:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Foreground reconstruction (Deformable-3DGS)
# ─────────────────────────────────────────────────────────────────────────────

def _do_reconstruct_foreground(scene_dir: Path, warm_init_path: str | None,
                                iterations: int):
    """Called inside the Modal container (or --local)."""
    import wandb

    colmap_dir = scene_dir / "_colmap_fg"
    prepare_colmap_structure(scene_dir, "fg_frames", colmap_dir)

    output_dir = scene_dir / "outputs" / "fg_gaussians"
    output_dir.mkdir(parents=True, exist_ok=True)

    # VERIFY: Deformable-3D-Gaussians train.py path and argument names.
    # The repo is at https://github.com/ingra14m/Deformable-3D-Gaussians
    # Check train.py for exact flag names if this fails.
    d3dgs_train = Path("/opt/deformable-3dgs/train.py")
    if not d3dgs_train.exists():
        raise FileNotFoundError(
            "/opt/deformable-3dgs/train.py not found."
        )

    cmd = [
        sys.executable, str(d3dgs_train),
        "-s", str(colmap_dir),
        "-m", str(output_dir),
        "--iterations",    str(iterations),
        "--densify_until_iter", str(min(iterations // 2, 15000)),
        "--mask_dir",      str(scene_dir / "fg_masks"),
    ]

    if warm_init_path:
        # Pass warm weights path so Deformable-3DGS loads them for deform MLP init
        # VERIFY: the exact flag name in Deformable-3DGS — may need to patch train.py
        cmd += ["--warm_init_path", warm_init_path]

    run_name = (
        f"fg_deform_warm_{scene_dir.name}"
        if warm_init_path
        else f"fg_deform_baseline_{scene_dir.name}"
    )

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "4drecon"),
            name=run_name,
            config={
                "iterations": iterations,
                "warm_init":  warm_init_path is not None,
                "mode":       "foreground",
            },
        )

    print(f"[Stage 3] Launching Deformable-3DGS:\n  {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    print(f"[Stage 3] Foreground reconstruction done → {output_dir}")

    if wandb_key:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Modal remote functions
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=training_image,
    gpu="A100",
    volumes=VOLUME_MAP,
    timeout=7200,
    memory=40960,
    secrets=[
        modal.Secret.from_name("wandb-secret", required=False),
    ],
)
def reconstruct_background(scene_dir_relative: str):
    """
    Modal function: 3DGS background reconstruction.
    Cost estimate: ~1.5 A100-hours ($2.25)
    """
    sd = Path(DATA_PATH) / scene_dir_relative
    _do_reconstruct_background(sd)
    data_volume.commit()


@app.function(
    image=training_image,
    gpu="A100",
    volumes=VOLUME_MAP,
    timeout=7200,
    memory=40960,
    secrets=[
        modal.Secret.from_name("wandb-secret", required=False),
    ],
)
def reconstruct_foreground(scene_dir_relative: str,
                            warm_init_path: str | None = None,
                            iterations: int = 30000):
    """
    Modal function: Deformable-3DGS foreground reconstruction.
    Cost estimate:
      Baseline (30k iter):  ~1.5 A100-hours ($2.25)
      Warm-init (500 iter): ~0.025 A100-hours ($0.04)
    """
    sd = Path(DATA_PATH) / scene_dir_relative
    _do_reconstruct_foreground(sd, warm_init_path, iterations)
    data_volume.commit()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: 3DGS background + Deformable-3DGS foreground reconstruction."
    )
    parser.add_argument("--scene-dir", required=True,
                        help="Scene directory (output of Stages 0-2)")
    parser.add_argument("--mode", choices=["bg", "fg", "both"], default="both",
                        help="What to reconstruct (default: both)")
    parser.add_argument("--warm-init", default=None,
                        help="Path to warm_deform_weights.pt (optional, FG only)")
    parser.add_argument("--fg-iterations", type=int, default=30000,
                        help="Deformable-3DGS iterations (default: 30000; "
                             "use 500 with --warm-init)")
    parser.add_argument("--local", action="store_true",
                        help="Run locally without Modal")
    args = parser.parse_args()

    sd = Path(args.scene_dir)

    if args.local:
        if args.mode in ("bg", "both"):
            _do_reconstruct_background(sd)
        if args.mode in ("fg", "both"):
            _do_reconstruct_foreground(sd, args.warm_init, args.fg_iterations)
    else:
        scene_name = sd.name
        print(f"[Stage 3] Launching Modal jobs for scene '{scene_name}' ...")
        with modal.enable_output():
            with app.run():
                if args.mode in ("bg", "both"):
                    reconstruct_background.remote(scene_name)
                if args.mode in ("fg", "both"):
                    reconstruct_foreground.remote(
                        scene_name,
                        args.warm_init,
                        args.fg_iterations,
                    )

    print("[Stage 3] DONE")


if __name__ == "__main__":
    main()

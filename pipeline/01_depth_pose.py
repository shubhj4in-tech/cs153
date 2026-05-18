"""
Stage 1 — Depth Estimation + Camera Pose Recovery
==================================================
Runs on Modal (A10G GPU).

Two sub-tasks:
  1. Apple Depth-Pro  → per-frame metric depth maps (.npy) + colorized PNGs
  2. DUSt3R           → camera poses (cameras.json, COLMAP-compatible)
                        + initial sparse point cloud (pointcloud.ply)

Approximate GPU cost: ~0.15 A10G-hours for a 30-frame scene.

Usage (local → triggers Modal run):
    python pipeline/01_depth_pose.py \\
        --scene-dir path/to/scene_dir/ \\
        [--model-dir /models]          # path inside Modal volume

    # or run local (no Modal) for debugging:
    python pipeline/01_depth_pose.py --local --scene-dir ./scene/
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root so modal_config is importable locally AND in Modal container
sys.path.insert(0, str(Path(__file__).parent.parent))

import modal
from modal_config import app, pipeline_image, VOLUME_MAP, MODELS_PATH, DATA_PATH, data_volume


# ─────────────────────────────────────────────────────────────────────────────
# Local implementation (runs inside Modal container OR standalone with --local)
# ─────────────────────────────────────────────────────────────────────────────

def run_depth_pro(frames_dir: Path, depth_dir: Path, model_dir: Path):
    """
    Run Apple Depth-Pro on all frames.

    Saves:
        depth_dir/<name>.npy          — float32 metric depth (metres)
        depth_dir/<name>_color.png    — jet-colourised depth for inspection
        depth_dir/focals.json         — predicted focal length per frame
    """
    import numpy as np
    import torch
    from PIL import Image
    import cv2

    depth_dir.mkdir(parents=True, exist_ok=True)

    import depth_pro

    print("[Stage 1] Loading Depth-Pro ...")
    # depth-pro pip package downloads checkpoint automatically to ~/.cache on first run
    # Pass HF_TOKEN if needed for gated model
    model, transform = depth_pro.create_model_and_transforms(
        device=torch.device("cuda"),
        precision=torch.half,
    )
    model.eval()

    frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    focals = {}

    for i, fp in enumerate(frame_files):
        print(f"  Depth-Pro [{i+1}/{len(frame_files)}] {fp.name}", end="\r")
        image = Image.open(fp).convert("RGB")
        # transform returns (C,H,W) tensor + f_px if known
        img_tensor = transform(image)

        with torch.no_grad():
            prediction = model.infer(img_tensor)

        depth_np = prediction["depth"].squeeze().cpu().float().numpy()  # (H, W)
        f_px     = float(prediction.get("focallength_px", 0.0))

        # Save .npy
        stem = fp.stem
        np.save(str(depth_dir / f"{stem}.npy"), depth_np)
        focals[stem] = f_px

        # Save colourised PNG
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(depth_dir / f"{stem}_color.png"), depth_color)

    (depth_dir / "focals.json").write_text(json.dumps(focals, indent=2))
    print(f"\n[Stage 1] Depth-Pro done — {len(frame_files)} frames saved to {depth_dir}")
    return focals


def run_dust3r(frames_dir: Path, depth_dir: Path, out_dir: Path, model_dir: Path,
               focals: dict) -> dict:
    """
    Run DUSt3R global alignment on all frames to recover camera poses and a
    sparse point cloud.

    Saves:
        out_dir/cameras.json       — COLMAP-like cameras + extrinsics
        out_dir/pointcloud.ply     — sparse point cloud from depth unprojection
    """
    import numpy as np
    import torch
    from plyfile import PlyData, PlyElement

    print("[Stage 1] Loading DUSt3R ...")

    # VERIFY: DUSt3R import path — adjust if the package installs differently
    from dust3r.inference   import inference
    from dust3r.model       import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt   import global_aligner, GlobalAlignerMode

    device = torch.device("cuda")

    # Prefer local checkpoint; fall back to HF hub
    ckpt_candidates = [
        model_dir / "dust3r" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",    # HF hub id
    ]
    ckpt = next((str(c) for c in ckpt_candidates
                 if isinstance(c, str) or c.exists()), ckpt_candidates[-1])

    model = AsymmetricCroCo3DStereo.from_pretrained(ckpt).to(device)
    model.eval()

    frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    img_paths   = [str(f) for f in frame_files]
    print(f"  Running DUSt3R on {len(img_paths)} frames ...")

    # Load + resize to 512 (DUSt3R's training size)
    images = load_images(img_paths, size=512)

    # Build all pairwise combinations (complete graph — fine for ≤60 frames)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    output = inference(pairs, model, device, batch_size=1, verbose=False)

    # Global point-cloud optimiser
    scene = global_aligner(
        output, device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
    )
    scene.compute_global_alignment(
        init="mst", niter=300, schedule="cosine", lr=0.01
    )

    poses   = scene.get_im_poses()    # (N, 4, 4) cam-to-world
    focals_dust3r = scene.get_focals()  # (N,)
    pts3d   = scene.get_pts3d()        # list of (H*W, 3) per view
    conf    = scene.get_conf()         # list of (H, W)

    N = len(frame_files)
    cameras = []
    all_pts = []
    all_cols = []

    # Import images for colour
    from PIL import Image as PILImage

    for i, (fp, pose, f_px_dust3r) in enumerate(
        zip(frame_files, poses, focals_dust3r)
    ):
        pose_np = pose.cpu().numpy().tolist()
        f_val   = float(focals.get(fp.stem, float(f_px_dust3r)))

        # Read image for point colours
        img = np.array(PILImage.open(fp).convert("RGB"))

        cam = {
            "id":    i,
            "name":  fp.name,
            "model": "PINHOLE",
            "width":  img.shape[1],
            "height": img.shape[0],
            "params": [f_val, f_val, img.shape[1] / 2, img.shape[0] / 2],
            "cam_to_world": pose_np,
        }
        cameras.append(cam)

        # Collect dense points (subsample to keep ply manageable)
        pts = pts3d[i].cpu().numpy()       # (H*W, 3)
        conf_mask = conf[i].cpu().numpy().flatten() > 1.5
        pts_filt  = pts[conf_mask]
        col_filt  = img.reshape(-1, 3)[conf_mask]

        # Subsample: keep at most 5000 pts per frame
        if len(pts_filt) > 5000:
            idx = np.random.choice(len(pts_filt), 5000, replace=False)
            pts_filt = pts_filt[idx]
            col_filt = col_filt[idx]

        all_pts.append(pts_filt)
        all_cols.append(col_filt)

    (out_dir / "cameras.json").write_text(json.dumps(cameras, indent=2))
    print(f"  Saved {N} camera poses → cameras.json")

    # Write PLY
    pts_np  = np.concatenate(all_pts,  axis=0).astype(np.float32)
    cols_np = np.concatenate(all_cols, axis=0).astype(np.uint8)
    vertex  = np.array(
        [(pts_np[j, 0], pts_np[j, 1], pts_np[j, 2],
          cols_np[j, 0], cols_np[j, 1], cols_np[j, 2])
         for j in range(len(pts_np))],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    el  = PlyElement.describe(vertex, "vertex")
    ply = PlyData([el], text=False)
    ply_path = out_dir / "pointcloud.ply"
    ply.write(str(ply_path))
    print(f"  Saved {len(pts_np)} points → pointcloud.ply")

    return {"n_cameras": N, "n_points": len(pts_np)}


def run_stage1_local(scene_dir: str, model_dir: str):
    """Entry-point when running without Modal (--local flag)."""
    sd = Path(scene_dir)
    frames_dir = sd / "frames"
    depth_dir  = sd / "depth"
    model_path = Path(model_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"frames/ not found in {sd}. Run Stage 0 first."
        )

    focals = run_depth_pro(frames_dir, depth_dir, model_path)
    stats  = run_dust3r(frames_dir, depth_dir, sd, model_path, focals)
    print(f"\n[Stage 1] DONE  cameras={stats['n_cameras']}  points={stats['n_points']}")


# ─────────────────────────────────────────────────────────────────────────────
# Modal remote function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=pipeline_image,
    gpu="A10G",
    volumes=VOLUME_MAP,
    timeout=3600,
    memory=24576,
)
def run_depth_and_pose_remote(scene_dir_relative: str):
    """
    Modal function: runs both Depth-Pro and DUSt3R on the uploaded scene.

    Cost estimate: ~0.15 A10G-hours ($0.04) for a 30-frame scene.

    Args:
        scene_dir_relative: path inside the data volume, e.g. "scenes/my_scene"
    """
    sd         = Path(DATA_PATH) / scene_dir_relative
    frames_dir = sd / "frames"
    depth_dir  = sd / "depth"
    model_path = Path(MODELS_PATH)

    focals = run_depth_pro(frames_dir, depth_dir, model_path)
    stats  = run_dust3r(frames_dir, depth_dir, sd, model_path, focals)

    # Flush volume so results are visible from outside
    data_volume.commit()
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Depth estimation (Depth-Pro) + camera pose (DUSt3R)."
    )
    parser.add_argument("--scene-dir",  required=True,
                        help="Local scene directory (output of Stage 0)")
    parser.add_argument("--model-dir",  default="./models",
                        help="Local path to model weights (default: ./models)")
    parser.add_argument("--local",      action="store_true",
                        help="Run locally without Modal (for debugging)")
    parser.add_argument("--scene-name", default=None,
                        help="Name used as relative path inside Modal volume "
                             "(default: basename of --scene-dir)")
    args = parser.parse_args()

    if args.local:
        print("[Stage 1] Running locally ...")
        run_stage1_local(args.scene_dir, args.model_dir)
    else:
        scene_name = args.scene_name or Path(args.scene_dir).name
        print(f"[Stage 1] Launching Modal job for scene '{scene_name}' ...")
        with modal.enable_output():
            with app.run():
                stats = run_depth_and_pose_remote.remote(scene_name)
        print(f"[Stage 1] DONE  {stats}")


if __name__ == "__main__":
    main()

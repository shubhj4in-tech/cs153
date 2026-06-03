"""
Stage 2 — Scene Separation (SAM2 + RAFT)
=========================================
Separates moving foreground from static background.

MANUAL STEP REQUIRED:
    Before running the automatic propagation, you must click the
    foreground object in frame 0 to seed SAM2.

    Step 1 — Interactive seeding (run locally):
        python pipeline/02_segment.py --scene-dir ./scene/ --interactive

        A window opens on frame 0. Click the object you want to track.
        Right-click to mark background corrections. Press 'q' when done.
        This saves  scene_dir/mask_frame0.json.

    Step 2 — Full pipeline (can run on Modal):
        python pipeline/02_segment.py --scene-dir ./scene/

Outputs:
    bg_frames/      — PNG frames with foreground inpainted
    fg_frames/      — Cropped foreground regions only
    fg_masks/       — Binary masks (.png, white=foreground)
    flow_maps/      — RAFT optical flow (.npy, shape HxWx2, units=pixels)
    separation_meta.json

Approximate GPU cost: ~0.20 A10G-hours for a 30-second scene at 2fps.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import modal
from modal_config import app, pipeline_image, VOLUME_MAP, DATA_PATH, MODELS_PATH, data_volume


# ─────────────────────────────────────────────────────────────────────────────
# Interactive seeding (local only — needs a display)
# ─────────────────────────────────────────────────────────────────────────────

def interactive_seed(scene_dir: Path) -> dict:
    """
    Open frame 0 in an OpenCV window. User clicks to add positive points,
    right-clicks for negative (background) points. Press 'q' to finish.
    Returns the mask spec as a dict and saves it to mask_frame0.json.
    """
    import cv2
    import numpy as np

    frames_dir = scene_dir / "frames"
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    frame0 = cv2.imread(str(frame_files[0]))
    display = frame0.copy()

    points  = []  # (x, y, label): label=1 pos, label=0 neg
    overlay = display.copy()

    def mouse_cb(event, x, y, flags, param):
        nonlocal overlay
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append({"x": x, "y": y, "label": 1})
            cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append({"x": x, "y": y, "label": 0})
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)

    cv2.namedWindow("SAM2 Seed — Left-click foreground, Right-click background, Q=done")
    cv2.setMouseCallback(
        "SAM2 Seed — Left-click foreground, Right-click background, Q=done",
        mouse_cb
    )

    print("[Stage 2] Interactive seeding:")
    print("  Left-click  = mark foreground points")
    print("  Right-click = mark background points")
    print("  Press 'q' when done")

    while True:
        cv2.imshow(
            "SAM2 Seed — Left-click foreground, Right-click background, Q=done",
            overlay
        )
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if not points:
        print("WARNING: No points selected. Saving empty mask spec.")

    spec = {
        "frame_index": 0,
        "frame_name":  frame_files[0].name,
        "points":      points,
    }
    spec_path = scene_dir / "mask_frame0.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"[Stage 2] Saved {len(points)} seed points → {spec_path}")
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# SAM2 mask propagation
# ─────────────────────────────────────────────────────────────────────────────

def propagate_masks(frames_dir: Path, mask_spec: dict, fg_masks_dir: Path,
                    model_dir: Path):
    """
    Use SAM2 video predictor to propagate the frame-0 seed across all frames.

    Saves fg_masks_dir/<frame_name>.png  (uint8, 0 or 255).
    """
    import numpy as np
    import torch
    from PIL import Image

    fg_masks_dir.mkdir(parents=True, exist_ok=True)

    # VERIFY: SAM2 install path and model config names.
    # See https://github.com/facebookresearch/sam2/blob/main/sam2/build_sam.py
    from sam2.build_sam import build_sam2_video_predictor

    ckpt_candidates = [
        model_dir / "sam2" / "sam2.1_hiera_large.pt",
        model_dir / "sam2" / "sam2_hiera_large.pt",
    ]
    ckpt = next((c for c in ckpt_candidates if c.exists()), None)
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        sam2_dir = model_dir / "sam2"
        sam2_dir.mkdir(parents=True, exist_ok=True)
        print("[Stage 2] Downloading SAM2 checkpoint from HF ...")
        hf_hub_download(
            repo_id="facebook/sam2.1-hiera-large",
            filename="sam2.1_hiera_large.pt",
            local_dir=str(sam2_dir),
        )
        ckpt = sam2_dir / "sam2.1_hiera_large.pt"

    # Config file distributed with the SAM2 package
    # SAM2 v2 (sam2.1) ships with updated config names
    cfg_candidates = ["configs/sam2.1/sam2.1_hiera_l.yaml", "sam2_hiera_l.yaml"]

    from hydra import initialize_config_dir
    import os, importlib
    cfg = cfg_candidates[0]   # SAM2 installed via pip uses this path

    print(f"[Stage 2] Loading SAM2 ...")
    predictor = build_sam2_video_predictor(cfg, str(ckpt), device="cuda")

    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.jpg"))

    # SAM2 requires JPEG files named as pure integers (e.g. 000001.jpg).
    import tempfile, shutil
    _tmp_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    for i, fp in enumerate(frame_files):
        jpeg_path = _tmp_dir / f"{i:07d}.jpg"
        Image.open(fp).convert("RGB").save(str(jpeg_path), quality=95)
    video_dir = str(_tmp_dir)
    print(f"[Stage 2] Prepared {len(frame_files)} frames for SAM2 in {_tmp_dir}")

    seed_frame_idx = mask_spec.get("frame_index", 0)

    points_xy = np.array(
        [[p["x"], p["y"]] for p in mask_spec["points"]], dtype=np.float32
    )
    labels = np.array(
        [p["label"] for p in mask_spec["points"]], dtype=np.int32
    )

    masks_by_idx = {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_path=video_dir)

        # Add seed points on the user-selected frame
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=seed_frame_idx,
            obj_id=1,
            points=points_xy,
            labels=labels,
        )

        # Propagate forward from seed frame
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=seed_frame_idx
        ):
            mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            masks_by_idx[frame_idx] = mask

        # Propagate backward to cover frames before the seed
        if seed_frame_idx > 0:
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=seed_frame_idx, reverse=True
            ):
                if frame_idx not in masks_by_idx:
                    mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                    masks_by_idx[frame_idx] = mask

    if _tmp_dir:
        shutil.rmtree(str(_tmp_dir), ignore_errors=True)

    # Use the first mask's shape for any missing frames
    ref_shape = next(iter(masks_by_idx.values())).shape if masks_by_idx else None
    for i, fp in enumerate(frame_files):
        if i in masks_by_idx:
            mask = masks_by_idx[i]
        elif ref_shape is not None:
            mask = np.zeros(ref_shape, dtype=np.uint8)
        else:
            img_h, img_w = np.array(Image.open(fp).convert("L")).shape
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
        Image.fromarray(mask).save(str(fg_masks_dir / fp.name))

    print(f"[Stage 2] SAM2 propagation done — {len(frame_files)} masks saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Foreground / background split + inpainting
# ─────────────────────────────────────────────────────────────────────────────

def split_fg_bg(frames_dir: Path, fg_masks_dir: Path,
                fg_frames_dir: Path, bg_frames_dir: Path):
    """
    For each frame:
      • bg_frames/<name>.png  — original frame with FG zeroed, then inpainted
                                using the median of surrounding frames
      • fg_frames/<name>.png  — tight crop of the foreground region

    The inpainting is simple frame-median: gather the N frames where a pixel is
    NOT under the foreground mask, and use their median value. For short videos
    some pixels may always be occluded — those fall back to nearest valid frame.
    """
    import numpy as np
    from PIL import Image
    import cv2

    fg_frames_dir.mkdir(parents=True, exist_ok=True)
    bg_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("*.png"))
    mask_files  = sorted(fg_masks_dir.glob("*.png"))

    # Load all frames and masks
    frames = [np.array(Image.open(f).convert("RGB")) for f in frame_files]
    masks  = [np.array(Image.open(m).convert("L")) > 127 for m in mask_files]

    H, W, _ = frames[0].shape
    N = len(frames)

    # Stack to (N, H, W, 3)
    stack = np.stack(frames, axis=0).astype(np.float32)  # (N, H, W, 3)
    msk   = np.stack(masks,  axis=0)                      # (N, H, W) bool

    # Compute background median: for each pixel, median over frames where mask=0
    # Vectorised: stack is (N, H, W, 3); msk is (N, H, W) bool.
    bg_mask_avail = ~msk   # True where pixel is background in that frame

    print("[Stage 2] Computing background median ...")
    # Use a masked sort: set foreground pixels to NaN then nanmedian.
    stack_masked = stack.copy()
    stack_masked[msk] = np.nan   # (N, H, W, 3)

    # nanmedian over the frame axis — O(N*H*W*3) but fully vectorised.
    bg_median = np.nanmedian(stack_masked, axis=0)   # (H, W, 3)

    # Any pixel always covered by foreground (all NaN) → use global mean fallback.
    always_fg = np.all(msk, axis=0)   # (H, W)
    if always_fg.any():
        global_mean = np.nanmean(stack_masked.reshape(N, -1, 3), axis=(0, 1))
        bg_median[always_fg] = global_mean

    bg_median = np.nan_to_num(bg_median, nan=0.0).astype(np.uint8)

    per_frame_meta = []
    for i, (fp, mask) in enumerate(zip(frame_files, masks)):
        frame = frames[i]

        # ── Background frame: replace FG region with median ───────────────
        bg_frame = frame.copy()
        bg_frame[mask] = bg_median[mask]
        Image.fromarray(bg_frame).save(str(bg_frames_dir / fp.name))

        # ── Foreground frame: crop tight bounding box ─────────────────────
        rows_with_fg = np.any(mask, axis=1)
        cols_with_fg = np.any(mask, axis=0)
        if rows_with_fg.any():
            r0, r1 = np.where(rows_with_fg)[0][[0, -1]]
            c0, c1 = np.where(cols_with_fg)[0][[0, -1]]
            # Add a small padding
            pad = 10
            r0 = max(0, r0 - pad); r1 = min(H, r1 + pad)
            c0 = max(0, c0 - pad); c1 = min(W, c1 + pad)
            fg_crop = frame[r0:r1, c0:c1]
            bbox = {"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)}
        else:
            fg_crop = np.zeros((1, 1, 3), dtype=np.uint8)
            bbox = None

        Image.fromarray(fg_crop).save(str(fg_frames_dir / fp.name))

        fg_px = int(mask.sum())
        per_frame_meta.append({
            "frame":   fp.name,
            "fg_pixels": fg_px,
            "bbox":    bbox,
        })

    print(f"[Stage 2] Split done: {N} frames → bg_frames/, fg_frames/")
    return per_frame_meta


# ─────────────────────────────────────────────────────────────────────────────
# RAFT optical flow
# ─────────────────────────────────────────────────────────────────────────────

def compute_flow(flow_frames_dir: Path, flow_maps_dir: Path):
    """
    Compute RAFT optical flow between consecutive full-fps frames.

    Uses torchvision's RAFT implementation (raft_large).
    Saves flow_maps/<name>.npy  — shape (H, W, 2), float32, units = pixels.
    """
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    import torchvision.transforms.functional as TF

    flow_maps_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Raft_Large_Weights.DEFAULT
    model   = raft_large(weights=weights).to(device).eval()
    # torchvision RAFT expects uint8 images in [0, 255]

    frame_files = sorted(flow_frames_dir.glob("*.png")) + \
                  sorted(flow_frames_dir.glob("*.jpg"))
    if len(frame_files) < 2:
        print("[Stage 2] WARNING: fewer than 2 flow frames — skipping flow.")
        return

    print(f"[Stage 2] Computing RAFT flow for {len(frame_files)-1} frame pairs ...")

    def load_tensor(fp):
        img = Image.open(fp).convert("RGB")
        t   = TF.to_tensor(img) * 255.0       # (3, H, W) float in [0,255]
        return t.unsqueeze(0).to(device)       # (1, 3, H, W)

    # RAFT requires H and W divisible by 8; pad if needed
    def pad8(t):
        """Pad (1, C, H, W) so H and W are multiples of 8."""
        _, _, h, w = t.shape
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            import torch.nn.functional as F
            t = F.pad(t, (0, pw, 0, ph))
        return t, h, w   # return original dims to crop output

    for i in range(len(frame_files) - 1):
        img1 = load_tensor(frame_files[i])
        img2 = load_tensor(frame_files[i + 1])
        img1, orig_h, orig_w = pad8(img1)
        img2, _, _           = pad8(img2)
        with torch.no_grad():
            # raft_large returns a list of flow predictions; last is the best
            flow_preds = model(img1, img2)
            flow = flow_preds[-1]              # (1, 2, H_pad, W_pad)

        flow_np = flow[0, :, :orig_h, :orig_w].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        stem    = frame_files[i].stem
        np.save(str(flow_maps_dir / f"{stem}_to_{frame_files[i+1].stem}.npy"),
                flow_np)

        print(f"  Flow [{i+1}/{len(frame_files)-1}]  "
              f"max_mag={float(np.abs(flow_np).max()):.1f}px", end="\r")

    print(f"\n[Stage 2] Flow done — {len(frame_files)-1} maps saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Combined Stage 2 entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(scene_dir: str, model_dir: str):
    sd = Path(scene_dir)
    frames_dir    = sd / "frames"
    flow_dir      = sd / "flow_frames"
    fg_masks_dir  = sd / "fg_masks"
    fg_frames_dir = sd / "fg_frames"
    bg_frames_dir = sd / "bg_frames"
    flow_maps_dir = sd / "flow_maps"
    model_path    = Path(model_dir)

    # Load mask spec (user must have run --interactive first)
    spec_path = sd / "mask_frame0.json"
    if not spec_path.exists():
        print(
            "\n[Stage 2] ERROR: mask_frame0.json not found.\n"
            "  Run interactively first:\n"
            f"    python pipeline/02_segment.py "
            f"--scene-dir {sd} --interactive\n"
            "  Then re-run without --interactive to continue."
        )
        sys.exit(1)

    mask_spec = json.loads(spec_path.read_text())

    propagate_masks(frames_dir, mask_spec, fg_masks_dir, model_path)
    per_frame_meta = split_fg_bg(frames_dir, fg_masks_dir,
                                  fg_frames_dir, bg_frames_dir)
    compute_flow(flow_dir, flow_maps_dir)

    meta = {
        "scene_dir":     str(sd),
        "n_frames":      len(per_frame_meta),
        "per_frame":     per_frame_meta,
    }
    (sd / "separation_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[Stage 2] DONE — separation_meta.json written")


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
def run_segmentation_remote(scene_dir_relative: str, model_dir_relative: str = ""):
    """
    Modal function: runs SAM2 propagation + RAFT on the scene.
    Cost estimate: ~0.20 A10G-hours ($0.05) for a 30-frame scene.
    Requires mask_frame0.json to already be present in the data volume.
    """
    sd         = Path(DATA_PATH) / scene_dir_relative
    model_path = Path(MODELS_PATH)
    run_stage2(str(sd), str(model_path))
    data_volume.commit()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: SAM2 mask propagation + RAFT optical flow."
    )
    parser.add_argument("--scene-dir",   required=True,
                        help="Scene directory (output of Stage 0)")
    parser.add_argument("--model-dir",   default="./models",
                        help="Path to model weights (default: ./models)")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive point-seeding UI (requires display)")
    parser.add_argument("--local",       action="store_true",
                        help="Run locally without Modal")
    args = parser.parse_args()

    if args.interactive:
        interactive_seed(Path(args.scene_dir))
        print("\n[Stage 2] Interactive seeding done.")
        print("  Re-run WITHOUT --interactive to propagate masks and compute flow.")
        sys.exit(0)

    if args.local:
        run_stage2(args.scene_dir, args.model_dir)
    else:
        scene_dir  = Path(args.scene_dir).resolve()
        scene_name = Path(args.scene_dir).name

        # Upload mask_frame0.json to Modal volume
        mask_json = scene_dir / "mask_frame0.json"
        if not mask_json.exists():
            print("[Stage 2] ERROR: mask_frame0.json not found. Run seed_ui first.")
            sys.exit(1)

        print(f"[Stage 2] Uploading seed spec and frames to Modal volume ...")
        subprocess.run(
            [sys.executable, "-m", "modal", "volume", "put", "--force",
             "4drecon-data", str(mask_json), f"{scene_name}/mask_frame0.json"],
            check=True,
        )
        for subdir in ("frames", "flow_frames"):
            local_subdir = scene_dir / subdir
            if local_subdir.exists():
                subprocess.run(
                    [sys.executable, "-m", "modal", "volume", "put", "--force",
                     "4drecon-data", str(local_subdir), f"{scene_name}/{subdir}"],
                    check=True,
                )

        print(f"[Stage 2] Launching Modal job for '{scene_name}' ...")
        with modal.enable_output():
            with app.run():
                run_segmentation_remote.remote(scene_name)

        # Download results
        import shutil
        print("[Stage 2] Downloading results ...")
        for subdir in ("fg_masks", "fg_frames", "bg_frames", "flow_maps",
                        "separation_meta.json"):
            local_path = scene_dir / subdir
            # Remove stale file/directory so modal can create the correct type
            if local_path.exists():
                if local_path.is_dir():
                    shutil.rmtree(str(local_path))
                else:
                    local_path.unlink()
            # Use trailing slash for remote dirs so modal downloads recursively
            is_file = subdir.endswith(".json")
            remote_path = f"{scene_name}/{subdir}" + ("" if is_file else "/")
            subprocess.run(
                [sys.executable, "-m", "modal", "volume", "get",
                 "--force", "4drecon-data", remote_path, str(local_path)],
                check=False,
            )
        print("[Stage 2] Done.")


if __name__ == "__main__":
    main()

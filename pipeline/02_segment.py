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
        model_dir / "sam2" / "sam2_hiera_large.pt",
        model_dir / "sam2" / "sam2.1_hiera_large.pt",
    ]
    ckpt = next((c for c in ckpt_candidates if c.exists()), None)
    if ckpt is None:
        # Try HF hub location used by download script
        ckpt = str(model_dir / "sam2" / "sam2_hiera_large.pt")
        print(f"  WARNING: SAM2 checkpoint not found at expected paths; "
              f"using {ckpt} (may fail)")

    # Config file distributed with the SAM2 package
    cfg = "sam2_hiera_l.yaml"   # VERIFY: exact yaml name in your SAM2 install

    print(f"[Stage 2] Loading SAM2 ({cfg}) ...")
    predictor = build_sam2_video_predictor(cfg, str(ckpt))

    frame_files = sorted(frames_dir.glob("*.png"))
    # SAM2 video predictor expects a directory of JPEG or PNG frames
    video_dir = str(frames_dir)

    points_xy = np.array(
        [[p["x"], p["y"]] for p in mask_spec["points"]], dtype=np.float32
    )
    labels = np.array(
        [p["label"] for p in mask_spec["points"]], dtype=np.int32
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_path=video_dir)

        # Add seed points on frame 0
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points_xy,
            labels=labels,
        )

        # Propagate forward through the video
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            # mask_logits: (n_objs, 1, H, W)
            mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            out_name = frame_files[frame_idx].name
            Image.fromarray(mask).save(str(fg_masks_dir / out_name))

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
    # To keep memory reasonable, process row by row
    bg_median = np.zeros((H, W, 3), dtype=np.float32)
    bg_mask_avail = ~msk                                   # True where bg is visible

    print("[Stage 2] Computing background median (this may take a moment) ...")
    for row in range(H):
        for ch in range(3):
            col_stack = stack[:, row, :, ch]    # (N, W)
            bg_vis    = bg_mask_avail[:, row, :] # (N, W)
            # For each column pixel, take median of visible bg values
            for col in range(W):
                vals = col_stack[bg_vis[:, col], col]
                if len(vals) > 0:
                    bg_median[row, col, ch] = np.median(vals)
                else:
                    bg_median[row, col, ch] = col_stack[:, col].mean()

    bg_median = bg_median.astype(np.uint8)

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

    for i in range(len(frame_files) - 1):
        img1 = load_tensor(frame_files[i])
        img2 = load_tensor(frame_files[i + 1])
        with torch.no_grad():
            # raft_large returns a list of flow predictions; last is the best
            flow_preds = model(img1, img2)
            flow = flow_preds[-1]              # (1, 2, H, W)

        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()   # (H, W, 2)
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
        scene_name = Path(args.scene_dir).name
        print(f"[Stage 2] Launching Modal job for '{scene_name}' ...")
        with modal.enable_output():
            with app.run():
                run_segmentation_remote.remote(scene_name)


if __name__ == "__main__":
    main()

"""
Stage 0 — Frame Extraction
==========================
Extracts frames from a phone video at two frame rates:
  • 2 fps  → frames/       (used by depth, pose, and reconstruction)
  • Full fps → flow_frames/ (used by RAFT optical flow computation)

Both outputs are resized to 960×540 for a good quality/compute tradeoff.

Usage:
    python pipeline/00_extract_frames.py \\
        --input path/to/video.mp4 \\
        --output path/to/scene_dir/

Outputs (inside --output dir):
    frames/          RGB PNGs at 2fps
    flow_frames/     RGB PNGs at original fps
    frames_meta.json metadata: timestamps, counts, video info
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import ffmpeg
import numpy as np


# Target resolution — matches DepthPro's sweet spot and stays < 1M pixels
TARGET_W = 960
TARGET_H = 540
RECON_FPS = 2        # frames used for 3DGS reconstruction
FLOW_SUBDIR = "flow_frames"
RECON_SUBDIR = "frames"


def probe_video(video_path: str) -> dict:
    """Return basic video metadata using ffprobe."""
    probe = ffmpeg.probe(video_path)
    video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
    if not video_streams:
        raise ValueError(f"No video stream found in {video_path}")
    vs = video_streams[0]

    # Parse frame rate fraction (e.g. "30/1" or "29970/1000")
    fps_str = vs.get("avg_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    duration = float(vs.get("duration", probe["format"].get("duration", 0)))

    return {
        "codec":    vs["codec_name"],
        "width":    int(vs["width"]),
        "height":   int(vs["height"]),
        "fps":      round(fps, 4),
        "duration": round(duration, 4),
        "total_frames": int(vs.get("nb_frames", round(fps * duration))),
    }


def extract_at_fps(video_path: str, out_dir: Path, fps: float,
                   width: int, height: int) -> list[dict]:
    """
    Extract frames from video_path at the given fps into out_dir.
    Returns list of {frame_index, timestamp_s, path} dicts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / "frame_%06d.png")

    (
        ffmpeg
        .input(video_path)
        .filter("fps", fps=fps)
        .filter("scale", width, height)
        .output(out_pattern, pix_fmt="rgb24")
        .overwrite_output()
        .run(quiet=True)
    )

    frame_files = sorted(out_dir.glob("frame_*.png"))
    records = []
    for i, fp in enumerate(frame_files):
        ts = round(i / fps, 6)
        records.append({
            "frame_index": i,
            "timestamp_s": ts,
            "path": str(fp.relative_to(out_dir.parent)),
        })
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: Extract frames from a phone video."
    )
    parser.add_argument("--input",  required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument(
        "--recon-fps", type=float, default=RECON_FPS,
        help=f"Frame rate for reconstruction frames (default: {RECON_FPS})"
    )
    parser.add_argument(
        "--width",  type=int, default=TARGET_W,
        help=f"Output frame width (default: {TARGET_W})"
    )
    parser.add_argument(
        "--height", type=int, default=TARGET_H,
        help=f"Output frame height (default: {TARGET_H})"
    )
    args = parser.parse_args()

    video_path = args.input
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Probe ─────────────────────────────────────────────────────────────────
    print(f"[Stage 0] Probing {video_path} ...")
    info = probe_video(video_path)
    print(
        f"  Codec: {info['codec']}  |  "
        f"Resolution: {info['width']}×{info['height']}  |  "
        f"FPS: {info['fps']}  |  "
        f"Duration: {info['duration']:.2f}s  |  "
        f"~{info['total_frames']} frames"
    )

    # ── Reconstruction frames (2 fps) ─────────────────────────────────────────
    recon_dir = out_dir / RECON_SUBDIR
    print(f"\n[Stage 0] Extracting reconstruction frames at {args.recon_fps} fps → {recon_dir}")
    recon_records = extract_at_fps(
        video_path, recon_dir, args.recon_fps, args.width, args.height
    )
    print(f"  Extracted {len(recon_records)} frames")

    # ── Full-fps frames for optical flow ──────────────────────────────────────
    flow_dir = out_dir / FLOW_SUBDIR
    print(f"\n[Stage 0] Extracting full-fps frames ({info['fps']} fps) → {flow_dir}")
    flow_records = extract_at_fps(
        video_path, flow_dir, info["fps"], args.width, args.height
    )
    print(f"  Extracted {len(flow_records)} frames")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "source_video": str(video_path),
        "output_dir":   str(out_dir),
        "target_resolution": {"width": args.width, "height": args.height},
        "video_info":   info,
        "recon_fps":    args.recon_fps,
        "recon_frame_count": len(recon_records),
        "flow_fps":     info["fps"],
        "flow_frame_count": len(flow_records),
        "recon_frames": recon_records,
        "flow_frames":  flow_records,
    }
    meta_path = out_dir / "frames_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"[Stage 0] DONE")
    print(f"  Reconstruction frames : {len(recon_records)} @ {args.recon_fps} fps → {recon_dir}/")
    print(f"  Optical-flow frames   : {len(flow_records)} @ {info['fps']} fps → {flow_dir}/")
    print(f"  Metadata              : {meta_path}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()

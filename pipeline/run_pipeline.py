"""
run_pipeline.py — Single-command 4D reconstruction pipeline
=============================================================

Runs all five stages in sequence, skipping stages whose outputs already exist.

Usage:
    python pipeline/run_pipeline.py \\
        --input  myvideo.mp4 \\
        --output ./my_scene/

Flags:
    --skip-stages 1 3    skip specific stage numbers
    --local              run everything locally (no Modal)
    --warm-init PATH     warm-init weights for Stage 3 FG reconstruction
    --viewer-dir PATH    where to write the .splat files (default: ./viewer/public/)
    --fg-iterations INT  Deformable-3DGS iterations (default 30000; use 500 with --warm-init)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Load .env automatically if it exists
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stamp():
    return time.strftime("%H:%M:%S")


def run_stage(stage_num: int, cmd: list[str], skip_if: Path | None = None):
    """Run a pipeline stage, skipping if skip_if path already exists."""
    if skip_if and skip_if.exists():
        print(f"[{_stamp()}] Stage {stage_num}: SKIPPED (outputs exist: {skip_if})")
        return

    print(f"\n[{_stamp()}] {'═'*55}")
    print(f"[{_stamp()}] Stage {stage_num}: STARTING")
    print(f"  Command: {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[{_stamp()}] Stage {stage_num}: FAILED (exit {result.returncode})")
        sys.exit(result.returncode)

    print(f"[{_stamp()}] Stage {stage_num}: DONE in {elapsed:.1f}s")
    return elapsed


def estimate_gpu_cost(elapsed_seconds: float, gpu_type: str = "A100") -> float:
    """Very rough cost estimate in USD."""
    hourly = {"A100": 1.50, "A10G": 0.30}.get(gpu_type, 1.00)
    return round(elapsed_seconds / 3600 * hourly, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Stage output sentinels (used to decide whether to skip)
# ─────────────────────────────────────────────────────────────────────────────

def sentinels(out: Path) -> dict[int, Path]:
    return {
        0: out / "frames_meta.json",
        1: out / "cameras.json",
        2: out / "separation_meta.json",
        3: out / "outputs" / "bg_gaussians",
        4: out / "viewer_output" / "manifest.json",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full 4D reconstruction pipeline end-to-end."
    )
    parser.add_argument("--input",   required=True,
                        help="Input video file (.mp4, .mov, etc.)")
    parser.add_argument("--output",  required=True,
                        help="Output scene directory")
    parser.add_argument("--skip-stages", nargs="*", type=int, default=[],
                        help="Stage numbers to skip (e.g. --skip-stages 0 1)")
    parser.add_argument("--local",   action="store_true",
                        help="Run all stages locally (no Modal)")
    parser.add_argument("--warm-init", default=None,
                        help="Path to warm_deform_weights.pt for foreground init")
    parser.add_argument("--viewer-dir", default=None,
                        help="Where to export .splat files "
                             "(default: <output>/viewer_output/)")
    parser.add_argument("--fg-iterations", type=int, default=None,
                        help="Deformable-3DGS iterations "
                             "(default: 500 with warm-init, 30000 without)")
    args = parser.parse_args()

    out       = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    viewer_dir = Path(args.viewer_dir) if args.viewer_dir else out / "viewer_output"

    skip = set(args.skip_stages)
    py   = sys.executable
    base = ["--local"] if args.local else []

    sens   = sentinels(out)
    times  = {}
    t_wall = time.time()

    # ── Stage 0: frame extraction ─────────────────────────────────────────────
    if 0 not in skip:
        t = run_stage(0, [
            py, "pipeline/00_extract_frames.py",
            "--input",  args.input,
            "--output", str(out),
        ], skip_if=sens[0])
        if t: times[0] = t

    # ── Stage 1: depth + pose ─────────────────────────────────────────────────
    if 1 not in skip:
        t = run_stage(1, [
            py, "pipeline/01_depth_pose.py",
            "--scene-dir", str(out),
            *base,
        ], skip_if=sens[1])
        if t: times[1] = t

    # ── Stage 2: segmentation ─────────────────────────────────────────────────
    mask_spec = out / "mask_frame0.json"
    if 2 not in skip:
        if not mask_spec.exists():
            print(
                f"\n[Pipeline] Stage 2 requires manual seeding.\n"
                f"  Run this first (needs a display):\n"
                f"\n"
                f"    python pipeline/02_segment.py "
                f"--scene-dir {out} --interactive\n"
                f"\n"
                f"  Then re-run run_pipeline.py to continue.\n"
            )
            sys.exit(0)

        t = run_stage(2, [
            py, "pipeline/02_segment.py",
            "--scene-dir", str(out),
            *base,
        ], skip_if=sens[2])
        if t: times[2] = t

    # ── Stage 3: reconstruction ────────────────────────────────────────────────
    if 3 not in skip:
        fg_iters = args.fg_iterations
        if fg_iters is None:
            fg_iters = 500 if args.warm_init else 30000

        stage3_cmd = [
            py, "pipeline/03_reconstruct.py",
            "--scene-dir",    str(out),
            "--mode",         "both",
            "--fg-iterations", str(fg_iters),
            *base,
        ]
        if args.warm_init:
            stage3_cmd += ["--warm-init", args.warm_init]

        t = run_stage(3, stage3_cmd, skip_if=sens[3])
        if t: times[3] = t

    # ── Stage 4: export ────────────────────────────────────────────────────────
    if 4 not in skip:
        t = run_stage(4, [
            py, "pipeline/04_export.py",
            "--scene-dir", str(out),
            "--output",    str(viewer_dir),
        ], skip_if=sens[4])
        if t: times[4] = t

    # ── Summary ───────────────────────────────────────────────────────────────
    total_wall = time.time() - t_wall
    total_gpu  = sum(times.values())

    print(f"\n{'═'*55}")
    print(f"  4D RECONSTRUCTION COMPLETE")
    print(f"{'─'*55}")
    print(f"  Input          : {args.input}")
    print(f"  Output         : {out}/")
    print(f"  Viewer assets  : {viewer_dir}/")
    print(f"  Wall time      : {total_wall/60:.1f} min")
    if total_gpu:
        print(f"  GPU time (est) : {total_gpu/60:.1f} min")
        cost = estimate_gpu_cost(total_gpu)
        print(f"  GPU cost (est) : ${cost:.2f}")
    print(f"{'─'*55}")
    print(f"\n  Open the viewer:")
    print(f"    cd viewer && npm run dev")
    print(f"    Copy {viewer_dir}/*.splat → viewer/public/\n")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()

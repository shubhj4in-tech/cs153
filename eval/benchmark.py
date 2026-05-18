"""
eval/benchmark.py — Baseline vs warm-init training time benchmark
===================================================================
Runs reconstruct_foreground() twice:
  (1) Baseline: random MLP initialisation, 30k iterations
  (2) Warm-init: pre-computed warm_deform_weights.pt, 500 iterations

Records:
  • Wall time (seconds)
  • PSNR at convergence (computed against held-out validation frames)

Outputs:
  eval/results/benchmark_results.json
  Prints a summary table

Usage:
    python eval/benchmark.py \\
        --scene-dir    ./my_scene/ \\
        --warm-init    ./outputs/warm_deform_weights.pt \\
        --gt-dir       ./my_scene/frames/            # GT frames for PSNR
        --output       ./eval/results/benchmark_results.json

Note: runs locally by default. Add --modal to run both on Modal A100s
(will incur cost, but gives clean comparison without local GPU interference).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def render_frames_from_checkpoint(ckpt_dir: Path, cameras_json: Path,
                                   output_dir: Path, n_frames: int = 20) -> bool:
    """
    Use the 3DGS render.py to produce rendered frames from a checkpoint.
    Returns True if successful.

    VERIFY: Deformable-3DGS render.py path and argument names.
    """
    render_dir = output_dir
    render_dir.mkdir(parents=True, exist_ok=True)

    render_script = Path("/opt/deformable-3dgs/render.py")
    if not render_script.exists():
        # Local fallback
        render_script = Path("./Deformable-3D-Gaussians/render.py")

    if not render_script.exists():
        print(f"  WARNING: render.py not found. Skipping PSNR computation.")
        return False

    cmd = [
        sys.executable, str(render_script),
        "-m", str(ckpt_dir),
        "--output", str(render_dir),
        "--skip_train",
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def run_single(
    scene_dir: Path,
    mode: str,        # "baseline" or "warm_init"
    warm_init_path: str | None,
    iterations: int,
    gt_dir: Path | None,
) -> dict:
    """
    Run one reconstruction, measure time, compute PSNR.
    Returns a results dict.
    """
    print(f"\n{'─'*50}")
    print(f"  Running: {mode}  ({iterations} iterations)")
    print(f"{'─'*50}")

    output_subdir = scene_dir / "outputs" / f"fg_{mode}"

    # Build the command
    cmd = [
        sys.executable, "pipeline/03_reconstruct.py",
        "--scene-dir",    str(scene_dir),
        "--mode",         "fg",
        "--fg-iterations", str(iterations),
        "--local",
    ]
    if warm_init_path:
        cmd += ["--warm-init", warm_init_path]

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    success = result.returncode == 0
    print(f"\n  Finished in {elapsed:.1f}s  success={success}")

    # ── Compute PSNR if GT is available ───────────────────────────────────────
    psnr_mean = None
    if success and gt_dir and gt_dir.exists():
        render_dir = scene_dir / "outputs" / f"render_{mode}"
        rendered   = render_frames_from_checkpoint(output_subdir, scene_dir / "cameras.json",
                                                    render_dir)
        if rendered:
            from eval.metrics import evaluate_directory
            metrics = evaluate_directory(str(render_dir), str(gt_dir))
            psnr_mean = metrics["psnr"]["mean"]
            print(f"  PSNR = {psnr_mean:.2f} dB")

    return {
        "mode":        mode,
        "iterations":  iterations,
        "elapsed_s":   round(elapsed, 2),
        "success":     success,
        "psnr_mean":   psnr_mean,
        "warm_init":   warm_init_path is not None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: random init vs warm-init reconstruction."
    )
    parser.add_argument("--scene-dir",   required=True,
                        help="Scene directory (must have outputs/bg_gaussians/ already)")
    parser.add_argument("--warm-init",   default=None,
                        help="Path to warm_deform_weights.pt")
    parser.add_argument("--gt-dir",      default=None,
                        help="Ground-truth frames for PSNR (optional)")
    parser.add_argument("--output",      default="./eval/results/benchmark_results.json",
                        help="Output JSON path")
    parser.add_argument("--baseline-iters", type=int, default=30000,
                        help="Iterations for baseline run (default: 30000)")
    parser.add_argument("--warm-iters",     type=int, default=500,
                        help="Iterations for warm-init run (default: 500)")
    parser.add_argument("--skip-baseline",  action="store_true",
                        help="Skip the baseline run (if already done)")
    args = parser.parse_args()

    sd     = Path(args.scene_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    results = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    if not args.skip_baseline:
        res_bl = run_single(sd, "baseline", None, args.baseline_iters, gt_dir)
        results.append(res_bl)
    else:
        print("  [Baseline] Skipped.")

    # ── Warm-init ─────────────────────────────────────────────────────────────
    if args.warm_init:
        res_wi = run_single(sd, "warm_init", args.warm_init, args.warm_iters, gt_dir)
        results.append(res_wi)
    else:
        print("  [Warm-init] --warm-init not provided, skipping.")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'─'*55}")
    header = f"{'Mode':<15}{'Iters':>8}{'Time (s)':>12}{'PSNR (dB)':>12}"
    print(f"  {header}")
    print(f"  {'─'*50}")
    for r in results:
        psnr_str = f"{r['psnr_mean']:.2f}" if r["psnr_mean"] is not None else "—"
        print(f"  {r['mode']:<15}{r['iterations']:>8}{r['elapsed_s']:>12.1f}{psnr_str:>12}")

    if len(results) == 2:
        speedup = results[0]["elapsed_s"] / max(results[1]["elapsed_s"], 1)
        print(f"\n  Speed-up: {speedup:.1f}×  "
              f"({results[0]['elapsed_s']:.0f}s → {results[1]['elapsed_s']:.0f}s)")

    print(f"{'═'*55}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "scene_dir": str(sd),
        "results":   results,
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()

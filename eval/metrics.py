"""
eval/metrics.py — Image quality metrics
=========================================
Functions:
    compute_psnr(img1, img2)     → float (dB)
    compute_ssim(img1, img2)     → float [0,1]
    compute_lpips(img1, img2)    → float (lower = better)

    evaluate_directory(pred_dir, gt_dir) → metrics.json

Usage:
    python eval/metrics.py \\
        --pred  ./outputs/rendered_frames/ \\
        --gt    ./data/ground_truth/ \\
        --output ./eval/results/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Args:
        img1, img2: (H, W, 3) uint8 or float32 in [0,255]

    Returns:
        PSNR in dB. ∞ if images are identical, lower bound ~0.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse  = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_val = 255.0
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray,
                  window_size: int = 11, sigma: float = 1.5) -> float:
    """
    Structural Similarity Index (SSIM).

    Uses scikit-image if available, otherwise a manual implementation.

    Returns value in [-1, 1]; 1.0 = perfect match.
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        img1_g = np.mean(img1.astype(np.float64), axis=-1)
        img2_g = np.mean(img2.astype(np.float64), axis=-1)
        return float(sk_ssim(img1_g, img2_g, data_range=255.0))
    except ImportError:
        pass

    # Manual SSIM (single-channel, simplified)
    import scipy.ndimage as ndi

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    x = img1.astype(np.float64).mean(axis=-1)
    y = img2.astype(np.float64).mean(axis=-1)

    def gauss_blur(arr, s=1.5):
        return ndi.gaussian_filter(arr, sigma=s)

    mu_x  = gauss_blur(x)
    mu_y  = gauss_blur(y)
    mu_xx = gauss_blur(x * x)
    mu_yy = gauss_blur(y * y)
    mu_xy = gauss_blur(x * y)

    sig_x  = mu_xx - mu_x ** 2
    sig_y  = mu_yy - mu_y ** 2
    sig_xy = mu_xy - mu_x * mu_y

    ssim_map = (
        (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
        / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2))
    )
    return float(ssim_map.mean())


def compute_lpips(img1: np.ndarray, img2: np.ndarray,
                   net: str = "alex") -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    Lower = more perceptually similar.

    Requires: pip install lpips torch

    Args:
        img1, img2: (H, W, 3) uint8 or float32 in [0, 255]
        net: 'alex' or 'vgg'
    """
    try:
        import torch
        import lpips
    except ImportError:
        print("  WARNING: lpips/torch not installed. Returning -1 for LPIPS.")
        return -1.0

    # Cache the LPIPS model per network type
    if not hasattr(compute_lpips, "_models"):
        compute_lpips._models = {}
    if net not in compute_lpips._models:
        compute_lpips._models[net] = lpips.LPIPS(net=net).eval()
        if torch.cuda.is_available():
            compute_lpips._models[net] = compute_lpips._models[net].cuda()

    model = compute_lpips._models[net]

    def to_tensor(arr):
        t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)  # → [-1,1]
        t = t.permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)
        if torch.cuda.is_available():
            t = t.cuda()
        return t

    with torch.no_grad():
        d = model(to_tensor(img1), to_tensor(img2))
    return float(d.item())


# ─────────────────────────────────────────────────────────────────────────────
# Directory evaluation
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load an image as (H, W, 3) uint8."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def evaluate_directory(
    pred_dir: str,
    gt_dir:   str,
    output_path: str | None = None,
) -> dict:
    """
    Compute PSNR, SSIM, LPIPS for all matching image pairs in two directories.

    Matches files by name. Skips pairs where GT is missing.
    """
    pred_dir = Path(pred_dir)
    gt_dir   = Path(gt_dir)

    pred_files = sorted(pred_dir.glob("*.png")) + sorted(pred_dir.glob("*.jpg"))
    gt_files   = {f.name: f for f in
                  list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg"))}

    results    = []
    psnr_vals  = []
    ssim_vals  = []
    lpips_vals = []

    n_matched = 0
    for pf in pred_files:
        gf = gt_files.get(pf.name)
        if gf is None:
            print(f"  No GT match for {pf.name} — skipping")
            continue

        pred = load_image(pf)
        gt   = load_image(gf)

        # Resize pred to match GT if needed
        if pred.shape[:2] != gt.shape[:2]:
            pred = np.array(
                Image.fromarray(pred).resize(
                    (gt.shape[1], gt.shape[0]), Image.BILINEAR
                )
            )

        p = compute_psnr(pred, gt)
        s = compute_ssim(pred, gt)
        l = compute_lpips(pred, gt)

        psnr_vals.append(p);  ssim_vals.append(s);  lpips_vals.append(l)
        results.append({"file": pf.name, "psnr": p, "ssim": s, "lpips": l})
        n_matched += 1
        print(f"  {pf.name}  PSNR={p:.2f}  SSIM={s:.4f}  LPIPS={l:.4f}")

    summary = {
        "n_frames": n_matched,
        "psnr":  {"mean": float(np.mean(psnr_vals))  if psnr_vals  else 0,
                  "std":  float(np.std(psnr_vals))   if psnr_vals  else 0},
        "ssim":  {"mean": float(np.mean(ssim_vals))  if ssim_vals  else 0,
                  "std":  float(np.std(ssim_vals))   if ssim_vals  else 0},
        "lpips": {"mean": float(np.mean(lpips_vals)) if lpips_vals else 0,
                  "std":  float(np.std(lpips_vals))  if lpips_vals else 0},
        "per_frame": results,
    }

    print(f"\nSummary ({n_matched} frames):")
    print(f"  PSNR  = {summary['psnr']['mean']:.2f} ± {summary['psnr']['std']:.2f} dB")
    print(f"  SSIM  = {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}")
    print(f"  LPIPS = {summary['lpips']['mean']:.4f} ± {summary['lpips']['std']:.4f}")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved → {out}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute PSNR / SSIM / LPIPS between rendered and GT frames."
    )
    parser.add_argument("--pred",   required=True,
                        help="Directory of predicted (rendered) frames")
    parser.add_argument("--gt",     required=True,
                        help="Directory of ground-truth frames")
    parser.add_argument("--output", default=None,
                        help="Path to save metrics.json (optional)")
    args = parser.parse_args()

    evaluate_directory(args.pred, args.gt, args.output)


if __name__ == "__main__":
    main()

"""
training/dataset.py — PyTorch Dataset for Blender synthetic data
=================================================================
Returns (rgb_clip, flow_clip) pairs for training the CogVideoX
flow-prediction head.

  rgb_clip:  (T, 3, H, W) float32 in [-1, 1]  (T=16 clips sampled from a scene)
  flow_clip: (T, 2, H, W) float32 in pixels    (from EXR Vector pass)

Usage:
    from training.dataset import BlenderFlowDataset, build_dataloader

    dl = build_dataloader("./data/synthetic/manifest.json")
    rgb, flow = next(iter(dl))   # (4, 16, 3, 256, 256), (4, 16, 2, 256, 256)
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

# VERIFY: imageio can read EXR if the OpenEXR Python binding is installed.
# If not available, install with: pip install imageio[freeimage]
# or use OpenCV: cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
import imageio


CLIP_LEN = 16         # temporal length fed to CogVideoX
TARGET_H = 256
TARGET_W = 256


class BlenderFlowDataset(Dataset):
    """
    Dataset over Blender synthetic scenes.

    Each __getitem__ samples a 16-frame clip from a random scene and
    returns aligned (rgb, flow) tensors.
    """

    def __init__(
        self,
        manifest_path: str,
        clip_len: int = CLIP_LEN,
        height: int   = TARGET_H,
        width: int    = TARGET_W,
        augment: bool = True,
    ):
        self.clip_len  = clip_len
        self.height    = height
        self.width     = width
        self.augment   = augment

        manifest = json.loads(Path(manifest_path).read_text())
        # Filter out scenes that errored
        self.scenes = [s for s in manifest if "error" not in s]

        if len(self.scenes) == 0:
            raise RuntimeError(
                f"No valid scenes found in {manifest_path}. "
                "Check that blender_render.py completed successfully."
            )

        # Pre-compute per-scene frame lists
        self._scene_frames: list[list[Path]] = []
        for sc in self.scenes:
            rdir = Path(sc["render_dir"])
            frames = sorted(rdir.glob("frame_*.png"))
            if len(frames) >= clip_len:
                self._scene_frames.append(frames)

        if len(self._scene_frames) == 0:
            raise RuntimeError(
                "No scene has enough rendered frames yet. "
                "Run blender_render.py first."
            )

    def __len__(self) -> int:
        return len(self._scene_frames)

    def __getitem__(self, idx: int):
        frames = self._scene_frames[idx]
        sc     = self.scenes[idx]
        flow_dir = Path(sc["flow_dir"])

        # Sample a contiguous clip of self.clip_len frames
        max_start = len(frames) - self.clip_len
        start     = random.randint(0, max_start)
        chosen    = frames[start : start + self.clip_len]

        # ── Augmentation parameters (shared across rgb+flow for consistency) ──
        do_hflip = self.augment and random.random() < 0.5

        # Color jitter params (rgb only)
        brightness = random.uniform(0.8, 1.2) if self.augment else 1.0
        contrast   = random.uniform(0.9, 1.1) if self.augment else 1.0
        saturation = random.uniform(0.9, 1.1) if self.augment else 1.0

        rgb_frames  = []
        flow_frames = []

        for fp in chosen:
            # ── RGB ──────────────────────────────────────────────────────────
            img = Image.open(fp).convert("RGB")
            img = img.resize((self.width, self.height), Image.BILINEAR)

            if self.augment:
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)
            if do_hflip:
                img = TF.hflip(img)

            rgb_t = TF.to_tensor(img)           # (3, H, W) in [0, 1]
            rgb_t = rgb_t * 2.0 - 1.0           # → [-1, 1]
            rgb_frames.append(rgb_t)

            # ── Flow ─────────────────────────────────────────────────────────
            flow_name = fp.stem + "0001.exr"    # Blender appends frame number
            flow_path = flow_dir / flow_name

            if flow_path.exists():
                flow_np = _load_flow_exr(flow_path)   # (H_orig, W_orig, 2)
            else:
                # Fallback: zero flow if EXR missing
                flow_np = np.zeros(
                    (img.height if hasattr(img, "height") else self.height,
                     img.width  if hasattr(img, "width")  else self.width, 2),
                    dtype=np.float32,
                )

            # Resize flow (scale magnitude proportionally)
            orig_h, orig_w = flow_np.shape[:2]
            scale_x = self.width  / orig_w
            scale_y = self.height / orig_h
            flow_resized = _resize_flow(flow_np, self.height, self.width,
                                         scale_x, scale_y)

            flow_t = torch.from_numpy(
                flow_resized.transpose(2, 0, 1)
            )  # (2, H, W)

            if do_hflip:
                flow_t        = TF.hflip(flow_t)
                flow_t[0]     = -flow_t[0]      # flip x-component of flow

            flow_frames.append(flow_t)

        rgb_clip  = torch.stack(rgb_frames,  dim=0)   # (T, 3, H, W)
        flow_clip = torch.stack(flow_frames, dim=0)   # (T, 2, H, W)

        return rgb_clip, flow_clip


# ─────────────────────────────────────────────────────────────────────────────
# EXR helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_flow_exr(path: Path) -> np.ndarray:
    """
    Load a Blender Vector pass EXR and extract the 2D forward flow (u, v).

    Blender's Vector pass has 4 channels: (u_prev, v_prev, u_next, v_next)
    in screen-space, normalised to [-1, 1] in pixel coordinates relative
    to the image size. We return the forward flow (channels 2, 3) in pixels.

    VERIFY: Channel ordering may differ between Blender versions.
    """
    try:
        # imageio with freeimage plugin reads EXR
        img = imageio.imread(str(path), format="EXR-FI")  # (H, W, C) float32
        if img.ndim == 3 and img.shape[2] >= 4:
            # Channels: X (prev_u), Y (prev_v), Z (next_u), W (next_v)
            u = img[:, :, 2]
            v = img[:, :, 3]
        elif img.ndim == 3 and img.shape[2] == 2:
            u, v = img[:, :, 0], img[:, :, 1]
        else:
            u = v = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        return np.stack([u, v], axis=-1).astype(np.float32)
    except Exception:
        # Fall back to OpenCV if imageio EXR support not available
        try:
            import cv2
            img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if img is None:
                raise ValueError("cv2 read returned None")
            if img.ndim == 3 and img.shape[2] >= 3:
                # OpenCV reads BGR; Blender Vector pass BGR = (v_next, u_next, *)
                u = img[:, :, 1].astype(np.float32)
                v = img[:, :, 0].astype(np.float32)
            else:
                u = v = np.zeros_like(img[:, :, 0], dtype=np.float32)
            return np.stack([u, v], axis=-1)
        except Exception:
            h, w = 256, 256
            return np.zeros((h, w, 2), dtype=np.float32)


def _resize_flow(flow: np.ndarray, th: int, tw: int,
                  scale_x: float, scale_y: float) -> np.ndarray:
    """Resize a (H, W, 2) flow field, scaling magnitude accordingly."""
    import cv2
    u = cv2.resize(flow[:, :, 0], (tw, th), interpolation=cv2.INTER_LINEAR) * scale_x
    v = cv2.resize(flow[:, :, 1], (tw, th), interpolation=cv2.INTER_LINEAR) * scale_y
    return np.stack([u, v], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    manifest_path: str,
    batch_size:  int  = 4,
    clip_len:    int  = CLIP_LEN,
    height:      int  = TARGET_H,
    width:       int  = TARGET_W,
    augment:     bool = True,
    num_workers: int  = 4,
) -> DataLoader:
    dataset = BlenderFlowDataset(
        manifest_path=manifest_path,
        clip_len=clip_len,
        height=height,
        width=width,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the BlenderFlowDataset.")
    parser.add_argument("--manifest", required=True,
                        help="Path to manifest.json from blender_render.py")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    dl = build_dataloader(args.manifest, batch_size=args.batch_size, num_workers=0)
    rgb, flow = next(iter(dl))

    print(f"rgb  shape : {rgb.shape}   dtype: {rgb.dtype}   range: [{rgb.min():.2f}, {rgb.max():.2f}]")
    print(f"flow shape : {flow.shape}  dtype: {flow.dtype}  range: [{flow.min():.2f}, {flow.max():.2f}]")
    print(f"Dataset size: {len(dl.dataset)} scenes")

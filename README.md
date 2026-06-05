# 4D Scene Reconstruction

**CS153 Project — Stanford**

A complete end-to-end pipeline that turns a 30-second phone video into a navigable 4D scene (3D + time) viewable in a browser. Static backgrounds are reconstructed with 3D Gaussian Splatting; dynamic foreground objects are handled by Deformable-3DGS. The novel research contribution is a fine-tuned CogVideoX-5b model whose internal motion features warm-initialize the Deformable-3DGS deformation MLP, collapsing 30-minute foreground training into under 2 minutes.

## Demo

> **[▶ Watch Demo Video (488 MB MP4)](https://github.com/shubhj4in-tech/cs153/releases/download/v1.0/CS.153.mp4)**

---

## Prerequisites

- **Conda** (Miniconda or Anaconda)
- **CUDA 12.1** compatible GPU (A10G or A100 recommended for Modal jobs)
- **Node.js ≥ 18** (for the viewer)
- **Blender 3.6 LTS** (for synthetic data generation; download from blender.org)
- A **Modal account** (free tier available at modal.com)
- A **Weights & Biases account** (free at wandb.ai) — optional but recommended
- A **Hugging Face account** with access to `zai-org/CogVideoX-5b-I2V`

---

## Setup

```bash
git clone https://github.com/shubhj4in-tech/cs153.git 4drecon
cd 4drecon

# Copy env template and fill in API keys
cp .env.example .env
# edit .env: WANDB_API_KEY, HF_TOKEN

# One-command setup (≈ 20 min first run)
bash scripts/setup.sh
```

`setup.sh` creates the `4drecon` conda environment, installs all dependencies in order, clones required repos, builds the 3DGS CUDA extensions, and downloads model weights.

---

## Usage

### Full pipeline (one command)

```bash
conda activate 4drecon
python pipeline/run_pipeline.py \
    --input  my_video.mp4 \
    --output ./my_scene/
```

**Important:** Stage 2 will pause and ask you to run the interactive SAM2 seeding step before it can continue (see Stage 2 below).

### Stage-by-stage (dynamic scene with moving foreground)

```bash
# Stage 0: Extract frames (2fps reconstruction + full-fps flow)
python pipeline/00_extract_frames.py --input video.mp4 --output ./scene/

# Stage 1: Depth maps + camera poses  (runs on Modal A100-80GB)
python pipeline/01_depth_pose.py --scene-dir ./scene/
# Add --local to run on your own GPU

# Stage 2 — MANUAL STEP REQUIRED:
python pipeline/02_segment.py --scene-dir ./scene/ --interactive
# A window opens on frame 0. Click the moving object. Press Q.
# Then run the full segmentation:
python pipeline/02_segment.py --scene-dir ./scene/

# Stage 3: Reconstruction  (runs on Modal A100-80GB — ~$5)
# BG trains in 4 × 7500-iter chunks to avoid Modal GPU preemption.
# Resume from a specific chunk if the connection drops:
python pipeline/03_reconstruct.py --scene-dir ./scene/ --mode both
# To resume after a connection drop (e.g. completed chunks 1–2):
python pipeline/03_reconstruct.py --scene-dir ./scene/ --mode bg --bg-start-chunk 3

# Stage 4: Export to .splat files
python pipeline/04_export.py --scene-dir ./scene/ --output ./viewer/public/
```

### Static / orbit scene (no foreground segmentation needed)

For videos where the camera moves around a static object, skip Stage 2:

```bash
# Stage 0: Extract at 4fps for more training views
python pipeline/00_extract_frames.py --input orbit.mp4 --output ./scene/ --fps 4

# Stage 1: same as above
python pipeline/01_depth_pose.py --scene-dir ./scene/

# Stage 2: skip — copy frames/ to bg_frames/ manually
cp -r ./scene/frames/ ./scene/bg_frames/

# Stage 3: background only
python pipeline/03_reconstruct.py --scene-dir ./scene/ --mode bg

# Stage 4: BG-only export with optional opacity boost
python pipeline/04_export.py --scene-dir ./scene/ --output ./viewer/public/ \
    --no-fg --opacity-boost 1.5
```

---

## Pipeline stages explained

| Stage | Script | What it does |
|-------|--------|-------------|
| 0 | `00_extract_frames.py` | Splits video into 2fps reconstruction frames and full-fps optical-flow frames at 960×540 |
| 1 | `01_depth_pose.py` | Runs Apple Depth-Pro for metric depth, then DUSt3R for camera poses; outputs `cameras.json` and an initial point cloud |
| 2 | `02_segment.py` | SAM2 propagates your clicked mask across all frames; RAFT computes dense optical flow; produces `bg_frames/`, `fg_frames/`, `fg_masks/`, `flow_maps/`. **Skip with `--no-fg` for static scenes.** |
| 3 | `03_reconstruct.py` | 3DGS trains on the static background in 4 × 7500-iter chunks (avoids GPU preemption); Deformable-3DGS trains on the foreground (30k baseline or 500 with warm-init). Use `--mode bg` for static scenes. |
| 4 | `04_export.py` | Merges BG + time-deformed FG Gaussians into one `.splat` file per timestep, plus `manifest.json`. `--no-fg` for BG-only export; `--opacity-boost N` to compensate for undertrained opacity. |

---

## Novel contribution: CogVideoX warm initialisation

### How it works

1. **Blender synthetic data** (`training/blender_render.py`): Generate 200 scenes with ground-truth optical flow and depth — 4 object types (walking biped, bouncing sphere, rigid body, cloth).

2. **CogVideoX fine-tuning** (`training/finetune_cogvideo.py`): Add LoRA (r=16) to temporal attention layers 12–18 + a linear flow-prediction head. Train on Blender flow for 5000 steps.

3. **Feature extraction** (`training/extract_features.py`): Register forward hooks on blocks 12–18, run the scene video through with one inference step, sample features at each Gaussian's projected 2D location.

4. **Warm init** (`training/warm_init.py`): A small linear projection maps motion embeddings → deformation MLP first-layer weights. The result replaces random initialisation.

5. **Fast reconstruction** (`pipeline/03_reconstruct.py --warm-init`): With better starting weights, Deformable-3DGS converges in 500 iterations instead of 30,000.

### Running the training pipeline

```bash
# Step 1: Generate Blender data (needs Blender 3.6)
blender --background --python training/blender_render.py -- \
    --output-dir ./data/synthetic/ --n-scenes 200

# Step 2–4: All training on Modal (one command)
python training/train_modal.py all \
    --blender-manifest ./data/synthetic/manifest.json \
    --scene-dir        ./my_scene/

# Or individual steps:
python training/train_modal.py finetune --blender-manifest ./data/synthetic/manifest.json
python training/train_modal.py extract-features --scene-dir ./my_scene/ --lora-dir ./outputs/cogvideox_lora/final/
python training/train_modal.py warm-init --embeddings ./outputs/motion_embeddings.pt --blender-dir ./data/synthetic/
```

---

## Viewer

```bash
# Copy exported .splat files + manifest.json to viewer/public/
cp -r ./my_scene/viewer_output/* viewer/public/

# Start the dev server
cd viewer && npm install && npm run dev
# Opens at http://localhost:5173
```

### Controls

| Input | Action |
|-------|--------|
| Left-drag | Orbit |
| Scroll | Zoom |
| Right-drag | Pan |
| Space | Play / Pause |
| ← → | Step one frame |
| ↑ ↓ | Step 5 frames |
| Home / End | First / last frame |

---

## Evaluation

```bash
# Compute PSNR / SSIM / LPIPS
python eval/metrics.py \
    --pred ./outputs/render_warm_init/ \
    --gt   ./my_scene/frames/ \
    --output ./eval/results/metrics.json

# Full benchmark comparison (baseline vs warm-init)
python eval/benchmark.py \
    --scene-dir ./my_scene/ \
    --warm-init ./outputs/warm_deform_weights.pt \
    --output    ./eval/results/benchmark_results.json

# Visualise results
python eval/visualize.py \
    --benchmark         ./eval/results/benchmark_results.json \
    --pred-baseline     ./outputs/render_baseline/ \
    --pred-warm-init    ./outputs/render_warm_init/ \
    --gt                ./my_scene/frames/ \
    --output            ./eval/figures/
```

---

## GPU cost summary

| Step | GPU | Est. cost |
|------|-----|-----------|
| Stage 1 (depth + pose, ≤31 frames) | A100-80GB | $0.10 |
| Stage 2 (SAM2 + RAFT) | A10G | $0.05 |
| Stage 3 BG (3DGS 30k iter, 4 chunks) | A100-80GB | $2.50 |
| Stage 3 FG baseline (30k) | A100-80GB | $2.50 |
| Stage 3 FG warm-init (500) | A100-80GB | $0.04 |
| CogVideoX fine-tuning (5k steps) | A100 | $9.00 |
| Feature extraction | A10G | $0.09 |
| Warm-init projection training | T4 | $0.02 |
| **Total (warm-init path)** | | **~$14** |

> **Note:** Stage 1 was upgraded from A10G to A100-80GB to handle DUSt3R's O(n²) memory cost for ≥20 frames. Stage 3 BG runs as 4 sequential 7500-iter chunks to avoid the ~8-minute GPU preemption window on Modal.

---

## Project structure

```
4drecon/
  pipeline/          5-stage reconstruction pipeline
  training/          CogVideoX fine-tuning + warm-init
  viewer/            Three.js browser viewer
  eval/              PSNR/SSIM/LPIPS metrics + benchmark
  scripts/           setup.sh, download_models.sh, test_pipeline.sh
  modal_config.py    Shared Modal app + volumes + container images
  requirements.txt   Pinned Python dependencies
```

---

## Citations & Acknowledgements

This project builds on the following open-source work:

| Component | Paper / Repo |
|-----------|-------------|
| **3D Gaussian Splatting** | Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023. [Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) · [Code](https://github.com/graphdeco-inria/gaussian-splatting) |
| **Deformable-3DGS** | Yang et al., "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction", CVPR 2024. [Paper](https://arxiv.org/abs/2309.13101) · [Code](https://github.com/ingra14m/Deformable-3D-Gaussians) |
| **CogVideoX-5b-I2V** | Yang et al., "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer", 2024. [Paper](https://arxiv.org/abs/2408.06072) · [Model](https://huggingface.co/zai-org/CogVideoX-5b-I2V) |
| **Apple Depth Pro** | Bochkovskii et al., "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second", 2024. [Paper](https://arxiv.org/abs/2410.02073) · [Code](https://github.com/apple/ml-depth-pro) |
| **DUSt3R** | Wang et al., "DUSt3R: Geometric 3D Vision Made Easy", CVPR 2024. [Paper](https://arxiv.org/abs/2312.14132) · [Code](https://github.com/naver/dust3r) |
| **SAM 2** | Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024. [Paper](https://arxiv.org/abs/2408.00714) · [Code](https://github.com/facebookresearch/sam2) |
| **RAFT** | Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow", ECCV 2020. [Paper](https://arxiv.org/abs/2003.12039) · [torchvision impl](https://pytorch.org/vision/stable/models/raft.html) |
| **PEFT / LoRA** | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022. [Paper](https://arxiv.org/abs/2106.09685) · [Code](https://github.com/huggingface/peft) |
| **GaussianSplats3D viewer** | [mkkellogg/GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) |
| **Modal** | Cloud GPU infrastructure — [modal.com](https://modal.com) |

---

## AI Usage Disclosure

Large language models (Claude Sonnet 4.5 / Claude Code) were used as coding assistants throughout this project in the following ways:

- **Boilerplate generation**: initial scaffolding for pipeline stages, Modal function wrappers, and the dataset loader.
- **Debugging**: identifying and resolving shape mismatches in the VAE/latent pipeline and COLMAP text-format writer.
- **Code review**: checking correctness of the quaternion → rotation-matrix conversion, the `.splat` binary packing, and the RAFT optical-flow integration.
- **Documentation**: drafting docstrings and the README structure.

All AI-generated code was reviewed, tested, and often substantially modified by the authors. The core research idea (using CogVideoX temporal attention features to warm-initialise the Deformable-3DGS deformation MLP) is original work.

---

## External Resources & Links

- **GitHub Repository**: <https://github.com/shubhj4in-tech/cs153>
- **Modal (cloud GPU)**: <https://modal.com>
- **Weights & Biases**: <https://wandb.ai>
- **Hugging Face — CogVideoX-5b-I2V**: <https://huggingface.co/zai-org/CogVideoX-5b-I2V>
- **3D Gaussian Splatting project page**: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- **Deformable-3D-Gaussians**: <https://github.com/ingra14m/Deformable-3D-Gaussians>
- **Apple Depth Pro**: <https://github.com/apple/ml-depth-pro>
- **DUSt3R**: <https://github.com/naver/dust3r>
- **SAM 2**: <https://github.com/facebookresearch/sam2>
- **antimatter15/splat** (`.splat` format spec): <https://github.com/antimatter15/splat>

---

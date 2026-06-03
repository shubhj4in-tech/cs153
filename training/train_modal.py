"""
training/train_modal.py — Modal orchestrator for all training jobs
===================================================================
One entry-point to run the full training pipeline on Modal:
  1. Upload Blender synthetic data to Modal volume
  2. Fine-tune CogVideoX (finetune_cogvideo.py)
  3. Extract motion features (extract_features.py)
  4. Train warm initialisation (warm_init.py)

Also provides individual commands for each step.

Usage:
    # Full training pipeline
    python training/train_modal.py \\
        --blender-manifest ./data/synthetic/manifest.json \\
        --scene-dir        ./my_scene/

    # Fine-tuning only
    python training/train_modal.py finetune \\
        --blender-manifest ./data/synthetic/manifest.json

    # Feature extraction only (after fine-tuning)
    python training/train_modal.py extract-features \\
        --scene-dir ./my_scene/ \\
        --lora-dir  ./outputs/cogvideox_lora/final/

    # Warm init only (after feature extraction)
    python training/train_modal.py warm-init \\
        --embeddings    ./outputs/motion_embeddings.pt \\
        --blender-dir   ./data/synthetic/
"""

import argparse
import sys
from pathlib import Path

import modal
from modal_config import app, training_image, VOLUME_MAP, DATA_PATH, MODELS_PATH, data_volume


# ─────────────────────────────────────────────────────────────────────────────
# Modal functions (wrappers that call the training modules inside the container)
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=training_image,
    gpu="A100",
    volumes=VOLUME_MAP,
    timeout=43200,
    memory=80000,
    secrets=[
        modal.Secret.from_name("wandb-secret", required=False),
        modal.Secret.from_name("hf-secret",    required=False),
    ],
)
def modal_finetune(
    manifest_relative: str,
    model_id: str   = "zai-org/CogVideoX-5b-I2V",
    max_steps: int  = 5000,
    output_relative: str = "training/cogvideox_lora",
):
    """
    Run CogVideoX fine-tuning inside a Modal container.
    Cost estimate: ~6 A100-hours ($9.00) for 5000 steps.
    """
    import sys
    sys.path.insert(0, "/root/4drecon")  # VERIFY: adjust if code is mounted differently

    from training.finetune_cogvideo import train

    manifest_path = str(Path(DATA_PATH) / manifest_relative)
    output_dir    = str(Path(DATA_PATH) / output_relative)
    # If model weights are in the models volume, use local path
    model_path    = Path(MODELS_PATH) / "cogvideox"
    effective_model_id = str(model_path) if model_path.exists() else model_id

    train(
        manifest_path=manifest_path,
        model_id=effective_model_id,
        max_steps=max_steps,
        output_dir=output_dir,
    )
    data_volume.commit()


@app.function(
    image=training_image,
    gpu="A10G",
    volumes=VOLUME_MAP,
    timeout=3600,
    memory=32768,
    secrets=[
        modal.Secret.from_name("hf-secret", required=False),
    ],
)
def modal_extract_features(
    scene_relative: str,
    lora_relative:  str,
    output_relative: str = "outputs/motion_embeddings.pt",
    model_id: str   = "zai-org/CogVideoX-5b-I2V",
):
    """
    Run CogVideoX feature extraction inside a Modal container.
    Cost estimate: ~0.3 A10G-hours ($0.09).
    """
    import sys
    sys.path.insert(0, "/root/4drecon")

    from training.extract_features import extract_motion_embeddings

    model_path = Path(MODELS_PATH) / "cogvideox"
    effective_model_id = str(model_path) if model_path.exists() else model_id

    extract_motion_embeddings(
        scene_dir=str(Path(DATA_PATH) / scene_relative),
        lora_dir=str(Path(DATA_PATH) / lora_relative),
        output_path=str(Path(DATA_PATH) / output_relative),
        model_id=effective_model_id,
    )
    data_volume.commit()


@app.function(
    image=training_image,
    gpu="T4",    # warm-init projection is small; T4 suffices
    volumes=VOLUME_MAP,
    timeout=3600,
    memory=16384,
)
def modal_warm_init(
    embeddings_relative: str,
    manifest_relative:   str,
    output_relative: str = "outputs/warm_deform_weights.pt",
    epochs: int = 50,
):
    """
    Run warm-init projection training inside a Modal container.
    Cost estimate: ~0.1 T4-hours ($0.02).
    """
    import sys
    sys.path.insert(0, "/root/4drecon")

    from training.warm_init import train_projection

    train_projection(
        embeddings_path=str(Path(DATA_PATH) / embeddings_relative),
        blender_manifest=str(Path(DATA_PATH) / manifest_relative),
        output_path=str(Path(DATA_PATH) / output_relative),
        epochs=epochs,
    )
    data_volume.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Local entry-point helpers
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_volume(local_dir: str, volume_path: str):
    """Upload a local directory to the Modal data volume."""
    import subprocess
    print(f"Uploading {local_dir} → Modal volume:{volume_path} ...")
    result = subprocess.run([
        "modal", "volume", "put", "4drecon-data",
        local_dir, volume_path,
    ])
    if result.returncode != 0:
        raise RuntimeError("modal volume put failed")
    print("  Upload complete.")


def download_from_volume(volume_path: str, local_path: str):
    """Download from Modal data volume to local path."""
    import subprocess
    print(f"Downloading Modal volume:{volume_path} → {local_path} ...")
    result = subprocess.run([
        "modal", "volume", "get", "4drecon-data",
        volume_path, local_path,
    ])
    if result.returncode != 0:
        raise RuntimeError("modal volume get failed")
    print("  Download complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def cli():
    """
    Local entry-point.  Call with:
        python training/train_modal.py [subcommand] [args]
    """
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Launch training jobs on Modal."
    )
    sub = parser.add_subparsers(dest="cmd")

    # ── finetune ──────────────────────────────────────────────────────────────
    ft = sub.add_parser("finetune", help="Fine-tune CogVideoX")
    ft.add_argument("--blender-manifest", required=True)
    ft.add_argument("--max-steps",        type=int, default=5000)
    ft.add_argument("--model-id",         default="zai-org/CogVideoX-5b-I2V")

    # ── extract-features ──────────────────────────────────────────────────────
    ef = sub.add_parser("extract-features", help="Extract CogVideoX features")
    ef.add_argument("--scene-dir", required=True)
    ef.add_argument("--lora-dir",  required=True)
    ef.add_argument("--output",    default="./outputs/motion_embeddings.pt")
    ef.add_argument("--model-id",  default="zai-org/CogVideoX-5b-I2V")

    # ── warm-init ─────────────────────────────────────────────────────────────
    wi = sub.add_parser("warm-init", help="Train warm-init projection")
    wi.add_argument("--embeddings",   required=True)
    wi.add_argument("--blender-dir",  required=True)
    wi.add_argument("--output",       default="./outputs/warm_deform_weights.pt")
    wi.add_argument("--epochs",       type=int, default=50)

    # ── all (full pipeline) ───────────────────────────────────────────────────
    al = sub.add_parser("all", help="Run full training pipeline")
    al.add_argument("--blender-manifest", required=True)
    al.add_argument("--scene-dir",        required=True)
    al.add_argument("--model-id",         default="zai-org/CogVideoX-5b-I2V")
    al.add_argument("--max-steps",        type=int, default=5000)

    args = parser.parse_args()

    with app.run():
        if args.cmd == "finetune":
            manifest_name = Path(args.blender_manifest).name
            print(f"[train_modal] Launching CogVideoX fine-tuning ...")
            modal_finetune.remote(
                manifest_relative=f"synthetic/{manifest_name}",
                model_id=args.model_id,
                max_steps=args.max_steps,
            )

        elif args.cmd == "extract-features":
            print(f"[train_modal] Launching feature extraction ...")
            modal_extract_features.remote(
                scene_relative=Path(args.scene_dir).name,
                lora_relative=Path(args.lora_dir).name,
                output_relative=Path(args.output).name,
                model_id=args.model_id,
            )

        elif args.cmd == "warm-init":
            print(f"[train_modal] Launching warm-init training ...")
            modal_warm_init.remote(
                embeddings_relative=Path(args.embeddings).name,
                manifest_relative=Path(args.blender_dir).name + "/manifest.json",
                output_relative=Path(args.output).name,
                epochs=args.epochs,
            )

        elif args.cmd == "all":
            print("[train_modal] Step 1: Fine-tuning CogVideoX ...")
            modal_finetune.remote(
                manifest_relative="synthetic/manifest.json",
                model_id=args.model_id,
                max_steps=args.max_steps,
            )
            print("[train_modal] Step 2: Extracting motion features ...")
            modal_extract_features.remote(
                scene_relative=Path(args.scene_dir).name,
                lora_relative="training/cogvideox_lora/final",
                output_relative="outputs/motion_embeddings.pt",
                model_id=args.model_id,
            )
            print("[train_modal] Step 3: Training warm-init projection ...")
            modal_warm_init.remote(
                embeddings_relative="outputs/motion_embeddings.pt",
                manifest_relative="synthetic/manifest.json",
                output_relative="outputs/warm_deform_weights.pt",
            )
            print("[train_modal] All training steps submitted.")

        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()

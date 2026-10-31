"""Evaluate prior generation for the matched, small-data VAE diagnostics.

Only stochastic VAE checkpoints are eligible. The deterministic beta-zero
autoencoder runs diagnose reconstruction capacity and do not define a trained
normal prior, so they are deliberately excluded from FID/IS evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from evaluation import compute_ignite_metrics, sha256_file
from train import choose_device, make_transform
from vae import ConvVAE


CONDITIONS = ("vae_l1_beta05", "vae_hybrid_beta05", "vae_l1_beta005")


def load_diagnostic_model(checkpoint_path: Path) -> tuple[ConvVAE, dict]:
    """Load the fixed 128x128/128-D stochastic VAE diagnostic architecture."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("stage") != "vae" or config.get("deterministic_training_decode_mu") is not False:
        raise ValueError(f"Prior evaluation requires a stochastic VAE checkpoint: {checkpoint_path}")
    if config.get("target_beta", 0) <= 0:
        raise ValueError("Beta-zero autoencoder diagnostics are ineligible for prior evaluation")
    state = checkpoint["model_state_dict"]
    if state["fc_mu.bias"].numel() != 128:
        raise ValueError("Expected the diagnostic's fixed 128-dimensional latent space")
    model = ConvVAE(image_size=128, latent_dim=128)
    model.load_state_dict(state, strict=True)
    return model.eval(), config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "experiments/diagnostics_v1")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/chest_xray")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--check-only", action="store_true", help="Validate checkpoints/configuration on CPU only")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("--conditions must not contain duplicates")
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    # Fail before using the GPU if any requested checkpoint is absent or ineligible.
    for condition in args.conditions:
        path = args.experiment_root / condition / "checkpoint.pt"
        model, config = load_diagnostic_model(path)
        if config.get("condition") != condition:
            raise ValueError(f"Checkpoint condition mismatch: {path}")
        print(f"Validated {condition}: training_images={config['training_images']}; " f"steps={config['steps']}; beta={config['target_beta']}", flush=True,)
        del model
    if args.check_only:
        return

    dataset = ImageFolder(args.data_root / "test", transform=make_transform(128, training=False))
    if args.samples > len(dataset):
        parser.error(f"--samples cannot exceed the {len(dataset)} test images")
    device = choose_device() if args.device == "auto" else torch.device(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    reference_path = ROOT / "artifacts/evaluation_manifest.json"
    reference = (json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else None)

    for condition in args.conditions:
        directory = args.experiment_root / condition
        checkpoint_path = directory / "checkpoint.pt"
        checkpoint_hash = sha256_file(checkpoint_path)
        model, config = load_diagnostic_model(checkpoint_path)
        model.to(device)
        print(f"Evaluating {condition}: device={device}; samples={args.samples}", flush=True)
        result = compute_ignite_metrics(model, loader, device, args.samples, seed=args.seed)
        manifest = result.pop("evaluation_manifest")
        manifest.update({ "checkpoint_sha256": checkpoint_hash, "checkpoint_filename": checkpoint_path.name, "source_condition": condition, "image_size": 128, "latent_dim": 128, "decoder_upsampling": "nearest", "training_images": config["training_images"], "training_steps": config["steps"], "training_config": config, "device": str(device), })
        if reference is not None:
            fields = ("protocol_version", "real_indices", "real_files", "latent_sha256", "sample_count")
            matches = {field: manifest[field] == reference[field] for field in fields}
            manifest["matches_original_corrected_evaluation_inputs"] = matches
            if args.samples == 512 and args.seed == 285 and not all(matches.values()):
                raise RuntimeError("Default evaluation inputs differ from the corrected reference")
        result.update({ "condition": condition, "model_version": "matched-small-data-residual-vae", "training_images": config["training_images"], "training_steps": config["steps"], "target_beta": config["target_beta"], "reconstruction_weights": config["reconstruction_weights"], "checkpoint_sha256": checkpoint_hash, "evaluation_device": str(device), "evaluation_manifest_file": "generation_manifest.json", "comparison_scope": ( "These matched diagnostics use 1024 training images and 800 optimizer steps. " "They are not equal-training-budget comparisons against the original " "5216-image, 20-epoch model. Single seed; no confidence interval." ), })
        if sha256_file(checkpoint_path) != checkpoint_hash:
            raise RuntimeError(f"Checkpoint changed during evaluation: {condition}")
        (directory / "generation_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (directory / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        del model


if __name__ == "__main__":
    main()

"""Recompute corrected FID/IS from a checkpoint without retraining.

The checkpoint remains immutable; artifacts/metrics.json is authoritative for
corrected evaluation. Historical checkpoint-embedded scores are deprecated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from evaluation import compute_ignite_metrics, sha256_file
from train import choose_device, make_transform
from vae import ConvVAE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/vae_checkpoint.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/chest_xray"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = choose_device() if args.device == "auto" else torch.device(args.device)
    dataset = ImageFolder(args.data_root / "test", transform=make_transform(config["image_size"], training=False))
    if not 2 <= args.samples <= len(dataset):
        parser.error(f"--samples must be in [2,{len(dataset)}]")
    model = ConvVAE(config["image_size"], config["latent_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluation only: device={device}; samples={args.samples}; checkpoint_sha256={checkpoint_hash}", flush=True)
    result = compute_ignite_metrics(model, loader, device, args.samples, seed=args.seed)
    manifest = result.pop("evaluation_manifest")
    manifest["checkpoint_sha256"] = checkpoint_hash
    manifest["checkpoint_filename"] = args.checkpoint.name
    manifest["image_size"] = config["image_size"]
    manifest["latent_dim"] = config["latent_dim"]
    manifest["device"] = str(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = args.output_dir / "metrics.json"
    previous = (json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else checkpoint.get("metrics", {}).copy())
    if "evaluation_protocol" not in previous:
        legacy_path = args.output_dir / "metrics_legacy.json"
        if not legacy_path.exists():
            legacy_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    updated = {**checkpoint.get("metrics", {}), **result}
    updated.update({ "evaluation_device": str(device), "checkpoint_sha256": checkpoint_hash, "evaluation_manifest_file": "evaluation_manifest.json", "checkpoint_embedded_metrics": "Historical unnormalized scores; use this external metrics.json for corrected evaluation.", })
    # Preserve the original training device and training metadata in the merged file.
    metric_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if sha256_file(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint unexpectedly changed during evaluation")
    print(json.dumps(updated, indent=2), flush=True)


if __name__ == "__main__":
    main()

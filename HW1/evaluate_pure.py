"""Evaluate pure VAE checkpoints with fixed normal-prior draws and corrected FID/IS.

This evaluator imports only the project VAE and metric/data utilities.
It accepts the preserved pure initializer and explicitly tagged pure-VAE
continuations, and rejects the adversarial experimental checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent / ".torch-cache"))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from diagnostic_metrics import batch_reconstruction_metrics
from evaluation import compute_ignite_metrics, sha256_file
from train import choose_device, make_transform
from vae import ConvVAE


ROOT = Path(__file__).resolve().parent
DISPLAY_SEEDS = (285, 1285, 2285)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_pure_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    pure_continuation = checkpoint.get("model_type") == "pure_vae" and checkpoint.get("initial_checkpoint_sha256") == "a59171e78e059a24e6be1dcaa03265f321471851a573cc54943ed874d5c7483c"
    if not pure_continuation and sha256_file(path) != "a59171e78e059a24e6be1dcaa03265f321471851a573cc54943ed874d5c7483c":
        raise ValueError("Expected the verified pure-VAE initializer or a pure_vae continuation; adversarial checkpoints are excluded")
    config = checkpoint["config"]
    if config.get("architecture", "conv") == "spatial":
        from spatial_vae import SpatialVAE
        model = SpatialVAE(image_size=int(config.get("image_size", 128)), global_dim=int(config.get("global_latent_dim", 128)), local_channels=int(config.get("local_channels", 4)), local_size=int(config.get("local_size", 16)), posterior_mode=config.get("spatial_posterior_mode", "residual"))
        if model.latent_dim != config["latent_dim"]:
            raise ValueError("Spatial VAE latent dimensions differ from the saved configuration")
    elif config.get("architecture", "conv") == "conv":
        model = ConvVAE(image_size=int(config.get("image_size", 128)), latent_dim=int(config.get("latent_dim", 128)))
    else:
        raise ValueError("Unknown pure-VAE architecture")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def fixed_prior_samples(model: ConvVAE, count: int, seed: int, device: torch.device, batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 1 or batch_size < 1:
        raise ValueError("Prior count and batch size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(count, model.latent_dim, generator=generator, dtype=torch.float32)
    generated = torch.cat([model.decode(batch.to(device)).cpu() for batch in latents.split(batch_size)])
    if not torch.isfinite(generated).all().item() or generated.min().item() < 0 or generated.max().item() > 1:
        raise ValueError("Decoder outputs must be finite image tensors in [0,1]")
    return generated, latents


@torch.no_grad()
def reconstruction_monitor(model: ConvVAE, dataset: ImageFolder, indices: list[int], device: torch.device, batch_size: int) -> tuple[dict[str, float], torch.Tensor]:
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("Reconstruction-monitor indices must be nonempty, unique and in range")
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(287))
    totals: dict[str, float] = {}
    first_pairs = None
    for images, _ in loader:
        images = images.to(device)
        mu, _ = model.encode(images)
        reconstruction = model.decode(mu)
        for name, value in batch_reconstruction_metrics(images, reconstruction).items():
            totals[name] = totals.get(name, 0.0) + value * len(images)
        if first_pairs is None:
            count = min(8, len(images))
            first_pairs = torch.cat((images[:count].cpu(), reconstruction[:count].cpu()))
    return {name: value / len(indices) for name, value in totals.items()}, first_pairs


@torch.no_grad()
def nearest_training_examples(generated: torch.Tensor, dataset, training_indices: list[int], batch_size: int = 32) -> tuple[torch.Tensor, list[dict]]:
    """Find coarse pixel neighbors, then report full-resolution pair metrics.

    This catches conspicuous copying; pixel-neighbor distance cannot establish
    absence of memorization, anatomy, or medical correctness.
    """
    if generated.ndim != 4 or generated.shape[1] != 1 or len(generated) < 1:
        raise ValueError("Expected a nonempty N x 1 x H x W generated tensor")
    if not training_indices or len(training_indices) != len(set(training_indices)) or min(training_indices) < 0 or max(training_indices) >= len(dataset):
        raise ValueError("Training indices must be nonempty, unique and in range")
    generated = generated.detach().cpu()
    queries = F.interpolate(generated, size=(32, 32), mode="bilinear", align_corners=False, antialias=True).flatten(1)
    query_square = queries.square().mean(1, keepdim=True)
    best_distances = torch.full((len(generated),), float("inf"))
    best_indices = torch.full((len(generated),), -1, dtype=torch.long)
    loader = DataLoader(Subset(dataset, training_indices), batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(288))
    offset = 0
    for images, _ in loader:
        features = F.interpolate(images.cpu(), size=(32, 32), mode="bilinear", align_corners=False, antialias=True).flatten(1)
        distances = (query_square + features.square().mean(1).unsqueeze(0) - 2 * queries @ features.T / features.shape[1]).clamp_min(0)
        values, within_batch = distances.min(1)
        improved = values < best_distances
        selected = torch.tensor(training_indices[offset : offset + len(images)], dtype=torch.long)
        best_distances[improved] = values[improved]
        best_indices[improved] = selected[within_batch[improved]]
        offset += len(images)
    references = torch.stack([dataset[int(index)][0] for index in best_indices])
    pairs = torch.stack((generated, references), dim=1).flatten(0, 1)
    records = []
    for position, index in enumerate(best_indices.tolist()):
        metrics = batch_reconstruction_metrics(references[position : position + 1], generated[position : position + 1])
        filename = Path(dataset.samples[index][0]).relative_to(dataset.root).as_posix() if hasattr(dataset, "samples") and hasattr(dataset, "root") else f"dataset-index:{index}"
        records.append({"prior_position": position, "nearest_training_index": index, "nearest_training_file": filename, "selection_mse_at_32x32": float(best_distances[position]), "full_resolution_metrics": metrics})
    return pairs, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/chest_xray")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "experiments/diagnostics_v1/data_manifest.json")
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--grid-seeds", nargs="+", type=int, default=list(DISPLAY_SEEDS))
    parser.add_argument("--nearest-count", type=int, default=0)
    parser.add_argument("--skip-metrics", action="store_true", help="Generate visual/reconstruction monitors without official-test FID/IS")
    parser.add_argument("--skip-reconstruction", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.nearest_count < 0 or args.samples < 2:
        parser.error("Require positive batch size, nonnegative nearest count, and samples >= 2")
    if len(args.grid_seeds) != len(set(args.grid_seeds)):
        parser.error("Grid seeds must be unique")
    protected = (ROOT / "artifacts", ROOT / "artifacts_center_beta05", ROOT / "artifacts_baseline", ROOT / "artifacts_padding_beta025")
    if args.output_dir.resolve() in {path.resolve() for path in protected}:
        parser.error("Use a new pure-VAE experiment output directory; preserved artifacts are read-only")
    device = choose_device() if args.device == "auto" else torch.device(args.device)
    checkpoint_hash = sha256_file(args.checkpoint)
    model, checkpoint = load_pure_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_manifest = {"checkpoint_sha256": checkpoint_hash, "checkpoint_filename": args.checkpoint.name, "image_size": model.image_size, "latent_dim": model.latent_dim, "sampling": "Independent CPU float32 N(0,I), temperature 1, complete fixed grids without individual selection", "grid_count_per_seed": 25, "grids": [], "training_provenance": checkpoint.get("training_provenance", checkpoint.get("trainingprovenance", {})), "initial_checkpoint_sha256": checkpoint.get("initial_checkpoint_sha256"), "reconstruction_scope": "Train-seen diagnostic: the existing full-data warm-start checkpoint used all 5,216 official training images, including this monitor; do not call this held-out validation."}
    fixed_grids = {}
    for seed in args.grid_seeds:
        images, latents = fixed_prior_samples(model, 25, seed, device, args.batch_size)
        filename = f"generated_seed{seed}.png"
        save_image(images, args.output_dir / filename, nrow=5, padding=2)
        fixed_grids[seed] = images
        visual_manifest["grids"].append({"seed": seed, "filename": filename, "latent_shape": list(latents.shape), "latent_sha256": tensor_hash(latents)})
    if args.baseline_checkpoint is not None:
        baseline_hash = sha256_file(args.baseline_checkpoint)
        baseline, _ = load_pure_model(args.baseline_checkpoint, device)
        if baseline.image_size != model.image_size:
            raise ValueError("Baseline comparison requires matching image dimensions")
        comparisons = []
        for seed in args.grid_seeds:
            before, _ = fixed_prior_samples(baseline, 25, seed, device, args.batch_size)
            after = fixed_grids[seed]
            paired_grid = torch.cat((before.view(5, 5, *before.shape[1:]), after.view(5, 5, *after.shape[1:])), dim=1).flatten(0, 1)
            filename = f"prior_comparison_seed{seed}.png"
            save_image(paired_grid, args.output_dir / filename, nrow=10, padding=2)
            comparisons.append({"seed": seed, "filename": filename, "layout": "Each row: five baseline images on the left; five corresponding new-model images on the right"})
        visual_manifest["baseline_comparison"] = {"checkpoint_sha256": baseline_hash, "grids": comparisons, "baseline_latent_dim": baseline.latent_dim, "selected_latent_dim": model.latent_dim, "latent_pairing": "Identical latent draws" if baseline.latent_dim == model.latent_dim else "Same fixed seed with model-specific Gaussian latent dimensionality; the latent vectors are different"}
        del baseline
    train_dataset = ImageFolder(args.data_root / "train", transform=make_transform(model.image_size, training=False))
    if not args.skip_reconstruction:
        split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        indices = split_manifest["splits"]["validation"]["indices"]
        metrics, pairs = reconstruction_monitor(model, train_dataset, indices, device, args.batch_size)
        result = {"checkpoint_sha256": checkpoint_hash, "image_count": len(indices), "dataset_split": "official train", "monitor_scope": visual_manifest["reconstruction_scope"], "reconstruction": "decode(mu); no posterior noise", "metrics": metrics, "split_manifest": args.split_manifest.name, "indices": indices}
        write_json(args.output_dir / "reconstruction_metrics.json", result)
        save_image(pairs, args.output_dir / "reconstructions.png", nrow=min(8, args.batch_size), padding=2)
    if args.nearest_count:
        generated, latents = fixed_prior_samples(model, args.nearest_count, args.seed, device, args.batch_size)
        pairs, records = nearest_training_examples(generated, train_dataset, list(range(len(train_dataset))), max(args.batch_size, 32))
        save_image(pairs, args.output_dir / "nearest_training.png", nrow=8, padding=2)
        write_json(args.output_dir / "nearest_training.json", {"checkpoint_sha256": checkpoint_hash, "seed": args.seed, "latent_sha256": tensor_hash(latents), "reference_count": len(train_dataset), "layout": "Alternating generated image and its nearest training image; four pairs per row", "criterion": "Minimum mean squared pixel distance after antialiased bilinear resize to 32x32; full-resolution metrics reported for that selected pair", "limitation": "A copying sanity check, not proof of absence of memorization or anatomical/medical validity", "pairs": records})
    if not args.skip_metrics:
        dataset = ImageFolder(args.data_root / "test", transform=make_transform(model.image_size, training=False))
        if args.samples > len(dataset):
            parser.error(f"Samples exceed the {len(dataset)} official test images")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        result = compute_ignite_metrics(model, loader, device, args.samples, seed=args.seed)
        manifest = result.pop("evaluation_manifest")
        manifest.update({"checkpoint_sha256": checkpoint_hash, "checkpoint_filename": args.checkpoint.name, "image_size": model.image_size, "latent_dim": model.latent_dim, "device": str(device), "training_provenance": visual_manifest["training_provenance"]})
        reference_path = ROOT / "artifacts/evaluation_manifest.json"
        if reference_path.exists():
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            fields = ("protocol_version", "real_indices", "real_files", "latent_sha256", "sample_count")
            matches = {field: manifest[field] == reference[field] for field in fields}
            manifest["matches_original_corrected_evaluation_inputs"] = matches
            manifest["latent_comparison_note"] = "All latent vectors match the original global VAE" if model.latent_dim == 128 else f"The selected spatial VAE uses {model.latent_dim} standard-normal coordinates; real indices and Inception protocol match, while latent vectors have a different dimensionality"
            if args.seed == 285 and args.samples == 512 and model.latent_dim == 128 and not all(matches.values()):
                raise RuntimeError("Default pure-VAE evaluation does not match the original corrected inputs")
        result.update({"checkpoint_sha256": checkpoint_hash, "evaluation_device": str(device), "model_version": checkpoint.get("model_version", checkpoint.get("metrics", {}).get("model_version", "pure-residual-vae")), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "evaluation_manifest_file": "generation_manifest.json", "training_provenance": visual_manifest["training_provenance"], "comparison_limit": "Single seed, 512-sample estimate, historical official-test observations; no significance or medical-validity claim."})
        write_json(args.output_dir / "generation_metrics.json", result)
        write_json(args.output_dir / "generation_manifest.json", manifest)
        print(json.dumps({name: result[name] for name in ("fid", "inception_score", "metric_samples", "evaluation_protocol", "checkpoint_sha256", "model_version", "parameter_count")}, indent=2), flush=True)
    if sha256_file(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("The evaluated checkpoint changed during evaluation")
    write_json(args.output_dir / "visual_manifest.json", visual_manifest)
    print(f"Saved pure-VAE evaluation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()


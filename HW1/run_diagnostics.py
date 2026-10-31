"""Controlled overfit and VAE-loss experiments; no official test-set selection.

All conditions share image preprocessing, initialization, minibatch order, and
optimizer settings. The deterministic beta=0 runs test memorization capacity;
their normal-prior samples are deliberately not evaluated as generative models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from diagnostic_metrics import batch_reconstruction_metrics
from train import choose_device, make_transform
from vae import ConvVAE, vae_loss


ROOT = Path(__file__).resolve().parent
CONDITIONS = {
    "overfit_l1_beta0": {"stage": "overfit", "weights": [1.0, 0.0, 0.0], "beta": 0.0},
    "overfit_hybrid_beta0": {"stage": "overfit", "weights": [0.75, 0.05, 0.2], "beta": 0.0},
    "vae_l1_beta05": {"stage": "vae", "weights": [1.0, 0.0, 0.0], "beta": 0.5},
    "vae_hybrid_beta05": {"stage": "vae", "weights": [0.75, 0.05, 0.2], "beta": 0.5},
    "vae_l1_beta005": {"stage": "vae", "weights": [1.0, 0.0, 0.0], "beta": 0.05},
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def filename_group(path: str, label: int, classes: list[str]) -> str:
    name = Path(path).stem
    if classes[label] == "NORMAL":
        match = re.match(r"^(IM-\d+|NORMAL2-IM-\d+)(?:-|$)", name)
    else:
        match = re.match(r"^(person\d+)_", name)
    if match is None:
        raise ValueError(f"Cannot infer a conservative filename group: {path}")
    return f"{classes[label]}:{match.group(1)}"


def select_data(dataset: ImageFolder, seed: int) -> tuple[dict, dict[str, list[int]]]:
    group_by_index = [filename_group(path, label, dataset.classes) for path, label in dataset.samples]
    groups: dict[str, list[int]] = {}
    for index, group in enumerate(group_by_index):
        groups.setdefault(group, []).append(index)
    validation_indices: list[int] = []
    training_pool: list[int] = []
    for label, target in enumerate((66, 190)):
        keys = sorted(group for group in groups if group.startswith(dataset.classes[label] + ":"))
        random.Random(seed + label).shuffle(keys)
        counts = [0]
        for key in keys:
            counts.append(counts[-1] + len(groups[key]))
        count = min(range(1, len(counts)), key=lambda i: abs(counts[i] - target))
        selected = set(keys[:count])
        for key in keys:
            (validation_indices if key in selected else training_pool).extend(groups[key])

    # Stratify the training subset using the original roughly 1:3 class balance.
    training_indices: list[int] = []
    for label, count in enumerate((263, 761)):
        candidates = sorted(index for index in training_pool if dataset.targets[index] == label)
        random.Random(seed + 10 + label).shuffle(candidates)
        training_indices.extend(candidates[:count])
    random.Random(seed + 20).shuffle(training_indices)
    random.Random(seed + 21).shuffle(validation_indices)
    overfit_indices = [index for label, count in ((0, 16), (1, 48)) for index in
                       [i for i in training_indices if dataset.targets[i] == label][:count]]
    random.Random(seed + 22).shuffle(overfit_indices)

    train_groups = {group_by_index[i] for i in training_indices}
    validation_groups = {group_by_index[i] for i in validation_indices}
    if train_groups & validation_groups:
        raise RuntimeError("Filename-group leakage between training and validation")
    selections = {"train": training_indices, "validation": validation_indices, "overfit": overfit_indices}
    manifest = {
        "seed": seed,
        "source_split": "official train only; official test not used",
        "grouping": "Conservative filename-inferred groups, not verified patient metadata",
        "group_patterns": {"NORMAL": r"^(IM-\d+|NORMAL2-IM-\d+)(?:-|$)", "PNEUMONIA": r"^(person\d+)_"},
        "preprocessing": "grayscale; aspect-preserving bilinear antialiased resize; center crop128; [0,1]; no augmentation",
        "training_pool_size": len(training_pool),
        "group_overlap": [],
        "splits": {},
    }
    for name, indices in selections.items():
        manifest["splits"][name] = {
            "indices": indices,
            "files": [Path(dataset.samples[i][0]).relative_to(dataset.root).as_posix() for i in indices],
            "filename_groups": sorted({group_by_index[i] for i in indices}),
            "class_counts": {cls: sum(dataset.targets[i] == label for i in indices) for label, cls in enumerate(dataset.classes)},
            "count": len(indices),
        }
    return manifest, selections


def load_inputs(dataset: ImageFolder, indices: list[int]) -> torch.Tensor:
    return torch.stack([dataset[index][0] for index in indices])


def batch_schedule(count: int, steps: int, batch_size: int, seed: int) -> torch.Tensor:
    if count % batch_size:
        raise ValueError("Training set size must be divisible by batch size for matched batches")
    generator = torch.Generator().manual_seed(seed)
    epochs = math.ceil(steps * batch_size / count)
    return torch.cat([torch.randperm(count, generator=generator) for _ in range(epochs)]).view(-1, batch_size)[:steps]


@torch.no_grad()
def evaluate_reconstruction(model: ConvVAE, images: torch.Tensor, batch_size: int = 32) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    for batch in images.split(batch_size):
        mu, logvar = model.encode(batch)
        decoded = model.decode(mu)
        metrics = batch_reconstruction_metrics(batch, decoded)
        metrics["raw_kl"] = float((-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)).mean().cpu())
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * len(batch)
    return {name: value / len(images) for name, value in totals.items()}


@torch.no_grad()
def save_reconstruction(model: ConvVAE, images: torch.Tensor, destination: Path) -> None:
    model.eval()
    mu, _ = model.encode(images[:8])
    save_image(torch.cat((images[:8], model.decode(mu))).cpu(), destination, nrow=8, padding=2)


def run_condition(name: str, args, tensors: dict[str, torch.Tensor], initial_state: dict[str, torch.Tensor]) -> dict:
    definition = CONDITIONS[name]
    deterministic = definition["stage"] == "overfit"
    total_steps = args.overfit_steps if deterministic else args.vae_steps
    directory = args.output_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "checkpoint.pt").exists():
        raise FileExistsError(f"Refusing to overwrite completed condition: {directory}")

    images = tensors["overfit"] if deterministic else tensors["train"]
    evaluation_images = images if deterministic else tensors["validation"]
    batch_indices = batch_schedule(len(images), total_steps, args.batch_size, args.seed + 100)
    # Explicit CPU draws make posterior noise independent of evaluation and condition ordering.
    noise = torch.randn(total_steps, args.batch_size, 128, generator=torch.Generator().manual_seed(args.seed + 101))
    model = ConvVAE(128, 128).to(args.resolved_device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    weights = definition["weights"]
    configuration = {
        "condition": name,
        "stage": definition["stage"],
        "training_images": len(images),
        "evaluation_images": len(evaluation_images),
        "steps": total_steps,
        "batch_size": args.batch_size,
        "initial_learning_rate": args.learning_rate,
        "minimum_learning_rate": 1e-5,
        "optimizer": "AdamW; weight_decay1e-4; betas0.9,0.999; gradient_clip10",
        "reconstruction_weights": dict(zip(("l1", "bce", "edge"), weights, strict=True)),
        "target_beta": definition["beta"],
        "warmup_steps": 0 if deterministic else args.warmup_steps,
        "deterministic_training_decode_mu": deterministic,
        "deterministic_evaluation_decode_mu": True,
        "initial_state_sha256": state_hash(initial_state),
        "batch_schedule_sha256": tensor_hash(batch_indices),
        "posterior_noise_sha256": tensor_hash(noise) if not deterministic else None,
        "seed": args.seed,
        "device": str(args.resolved_device),
        "interpretation": "Different raw reconstruction scales also change effective KL/reconstruction balance at fixed beta; not an isolated edge-loss ablation.",
    }
    write_json(directory / "config.json", configuration)
    history = []
    start = time.perf_counter()
    print(f"START {name}: steps={total_steps}; train={len(images)}; evaluation={len(evaluation_images)}", flush=True)
    first_metrics = evaluate_reconstruction(model, evaluation_images)
    history.append({"step": 0, "evaluation": first_metrics, "elapsed_seconds": 0.0})
    running = torch.zeros(6, device=args.resolved_device)
    interval_count = 0
    for step in range(1, total_steps + 1):
        model.train()
        batch = images[batch_indices[step - 1].to(args.resolved_device)]
        optimizer.zero_grad(set_to_none=True)
        mu, logvar = model.encode(batch)
        beta = 0.0 if deterministic else definition["beta"] * min(1.0, step / max(1, args.warmup_steps))
        latent = mu if deterministic else mu + (0.5 * logvar).exp() * noise[step - 1].to(args.resolved_device)
        decoded = model.decode(latent)
        # The diagnostic uses mu and no KL computation: the log-variance head is not trained.
        loss_mu = mu if not deterministic else torch.zeros_like(mu)
        loss_logvar = logvar if not deterministic else torch.zeros_like(logvar)
        losses = vae_loss(decoded, batch, loss_mu, loss_logvar, beta, *weights)
        if not bool(torch.isfinite(losses[0]).item()):
            raise FloatingPointError(f"Non-finite objective in {name} at step{step}")
        losses[0].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        running += torch.stack([loss.detach() for loss in losses])
        interval_count += 1
        if step % args.log_every == 0 or step == total_steps:
            values = (running / interval_count).cpu().tolist()
            evaluation = evaluate_reconstruction(model, evaluation_images)
            row = {
                "step": step,
                "beta": beta,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training_interval": dict(zip(("objective", "reconstruction", "kl", "l1", "bce", "edge"), values, strict=True)),
                "evaluation": evaluation,
                "gradient_norm_before_clip": float(norm.detach().cpu()),
                "elapsed_seconds": time.perf_counter() - start,
            }
            history.append(row)
            write_json(directory / "history.json", history)
            save_reconstruction(model, evaluation_images, directory / "reconstructions.png")
            print(f"{name} step{step}/{total_steps}: MAE={evaluation['mae']:.5f}; SSIM={evaluation['ssim']:.4f}; PSNR={evaluation['psnr']:.2f}; beta={beta:.3f}; seconds={row['elapsed_seconds']:.1f}", flush=True)
            running.zero_()
            interval_count = 0

    final = {
        "condition": name,
        "configuration": configuration,
        "initial_reconstruction": first_metrics,
        "final_reconstruction": history[-1]["evaluation"],
        "training_seconds": time.perf_counter() - start,
        "evaluation_type": "deterministic decode(mu); overfit training images" if deterministic else "deterministic decode(mu); filename-group-separated internal validation",
    }
    if not deterministic:
        final["train_subset_reconstruction"] = evaluate_reconstruction(model, images)
        latent = torch.randn(25, 128, generator=torch.Generator().manual_seed(args.seed)).to(args.resolved_device)
        model.eval()
        with torch.no_grad():
            save_image(model.decode(latent).cpu(), directory / "generated_samples.png", nrow=5, padding=2)
    write_json(directory / "metrics.json", final)
    torch.save({"model_state_dict": model.state_dict(), "config": configuration}, directory / "checkpoint.pt")
    print(f"FINISHED {name}: {json.dumps(final['final_reconstruction'])}", flush=True)
    return final


def save_summary(output: Path) -> None:
    results = {}
    histories = {}
    for name in CONDITIONS:
        path = output / name / "metrics.json"
        if path.exists():
            results[name] = json.loads(path.read_text())
            histories[name] = json.loads((path.parent / "history.json").read_text())
    write_json(output / "comparison.json", results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, history in histories.items():
        label = name.replace("overfit_", "AE ").replace("vae_", "VAE ").replace("_", " ")
        axes[0].plot([row["step"] for row in history], [row["evaluation"]["mae"] for row in history], marker="o", ms=3, label=label)
        axes[1].plot([row["step"] for row in history], [row["evaluation"]["ssim"] for row in history], marker="o", ms=3, label=label)
    axes[0].set(xlabel="Optimizer step", ylabel="MAE / pixel", title="Deterministic reconstruction")
    axes[1].set(xlabel="Optimizer step", ylabel="SSIM", title="Deterministic reconstruction")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("AE: 64 training images; VAE: 255 validation images (different tasks)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output / "comparison_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/chest_xray")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/diagnostics_v1")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--overfit-steps", type=int, default=800)
    parser.add_argument("--vae-steps", type=int, default=800)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.overfit_steps, args.vae_steps, args.log_every) <= 0:
        parser.error("Step counts and log interval must be positive")
    args.resolved_device = choose_device() if args.device == "auto" else torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ImageFolder(args.data_root / "train", transform=make_transform(128, False))
    manifest, selections = select_data(dataset, args.seed)
    manifest_path = args.output_dir / "data_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output uses a different data split")
    write_json(manifest_path, manifest)
    print("Loading fixed tensors (no augmentation)...", flush=True)
    tensors = {name: load_inputs(dataset, indices).to(args.resolved_device) for name, indices in selections.items()}
    print("Split counts: " + str({name: len(value) for name, value in tensors.items()}), flush=True)
    torch.manual_seed(args.seed)
    initial_state = {name: value.detach().cpu().clone() for name, value in ConvVAE(128, 128).state_dict().items()}
    for name in args.conditions:
        run_condition(name, args, tensors, initial_state)
        save_summary(args.output_dir)
    print("DONE " + str(args.output_dir), flush=True)


if __name__ == "__main__":
    main()

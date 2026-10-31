"""Train and evaluate a residual VAE on the Kaggle chest X-ray dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from evaluation import compute_ignite_metrics
from vae import ConvVAE, vae_loss


@dataclass
class ExperimentConfig:
    data_root: str
    output_dir: str
    image_size: int = 128
    latent_dim: int = 128
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    beta: float = 0.5
    beta_warmup_epochs: int = 10
    l1_weight: float = 0.75
    bce_weight: float = 0.05
    edge_weight: float = 0.2
    eval_samples: int = 512
    eval_batch_size: int = 8
    num_workers: int = 0
    seed: int = 285
    max_train_samples: int | None = None
    skip_metrics: bool = False
    augment: bool = True


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/chest_xray")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--beta-warmup-epochs", type=int, default=10)
    parser.add_argument("--l1-weight", type=float, default=0.75)
    parser.add_argument("--bce-weight", type=float, default=0.05)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.set_defaults(augment=True)
    return ExperimentConfig(**vars(parser.parse_args()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_transform(image_size: int, training: bool) -> transforms.Compose:
    operations: list[object] = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True,),
        transforms.CenterCrop(image_size),
    ]
    if training:
        operations.extend([ transforms.RandomAffine( degrees=5, translate=(0.03, 0.03), scale=(0.98, 1.02), interpolation=transforms.InterpolationMode.BILINEAR, fill=0, ), transforms.ColorJitter(brightness=0.05, contrast=0.05), ])
    operations.append(transforms.ToTensor())
    return transforms.Compose(operations)


def maybe_subset(dataset: Dataset, maximum: int | None, seed: int) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].tolist()
    return Subset(dataset, indices)


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, dict]:
    root = Path(config.data_root)
    train_root, test_root = root / "train", root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"Expected {train_root} and {test_root}. Run download_data.py first.")
    train_transform = make_transform(config.image_size, training=config.augment)
    test_transform = make_transform(config.image_size, training=False)
    full_train = ImageFolder(train_root, transform=train_transform)
    test_dataset = ImageFolder(test_root, transform=test_transform)
    train_dataset = maybe_subset(full_train, config.max_train_samples, config.seed)

    loader_args = dict(num_workers=config.num_workers, pin_memory=False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed), **loader_args,)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed + 1), **loader_args,)
    metadata = {
        "train_images_available": len(full_train),
        "train_images_used": len(train_dataset),
        "test_images": len(test_dataset),
        "classes": full_train.classes,
        "train_class_counts": {
            full_train.classes[index]: full_train.targets.count(index)
            for index in range(len(full_train.classes))
        },
        "test_class_counts": {
            test_dataset.classes[index]: test_dataset.targets.count(index)
            for index in range(len(test_dataset.classes))
        },
        "preprocessing": "grayscale; aspect-preserving antialiased resize; center crop",
        "augmentation": ("rotation +/-5 deg; translation +/-3%; scale 0.98-1.02; brightness/contrast +/-5%" if config.augment else "none"),
    }
    return train_loader, test_loader, metadata


def run_epoch(model: ConvVAE, loader: DataLoader, device: torch.device, beta: float, l1_weight: float, bce_weight: float, edge_weight: float, optimizer: torch.optim.Optimizer | None = None,) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
        "l1": 0.0,
        "bce": 0.0,
        "edge": 0.0,
    }
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, _ in loader:
            images = images.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(images)
            loss, reconstruction_loss, kl_loss, l1_loss, bce_loss, edge_loss = vae_loss(reconstruction, images, mu, logvar, beta=beta, l1_weight=l1_weight, bce_weight=bce_weight, edge_weight=edge_weight,)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            batch_size = images.shape[0]
            values = (loss, reconstruction_loss, kl_loss, l1_loss, bce_loss, edge_loss)
            for key, value in zip(sums, values, strict=True):
                sums[key] += value.item() * batch_size
            count += batch_size
    return {key: value / count for key, value in sums.items()}


def beta_for_epoch(epoch: int, target_beta: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return target_beta
    return target_beta * min(1.0, epoch / warmup_epochs)


def save_loss_plot(history: list[dict[str, float]], path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], "o-", label="Train")
    axes[0].plot(epochs, [entry["test_loss"] for entry in history], "s-", label="Test")
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Loss / image")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [entry["train_reconstruction"] for entry in history], "o-", label="0.75 L1 + 0.05 BCE + 0.20 edge",)
    axes[1].plot(epochs, [entry["beta"] * entry["train_kl"] for entry in history], "s-", label="Weighted KL",)
    axes[1].set(title="Training components", xlabel="Epoch", ylabel="Loss / image")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_examples(model: ConvVAE, test_loader: DataLoader, output_dir: Path, device: torch.device) -> None:
    model.eval()
    torch.manual_seed(285)
    generated = model.sample(25, device).cpu()
    save_image(generated, output_dir / "generated_samples.png", nrow=5, padding=2)

    images, _ = next(iter(test_loader))
    images = images[:8].to(device)
    reconstruction, _, _ = model(images)
    comparison = torch.cat([images.cpu(), reconstruction.cpu()])
    save_image(comparison, output_dir / "reconstructions.png", nrow=8, padding=2)


def main() -> None:
    config = parse_args()
    if not np.isclose(config.l1_weight + config.bce_weight + config.edge_weight, 1.0):
        raise ValueError("--l1-weight, --bce-weight, and --edge-weight must sum to 1.0")
    set_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader, dataset_metadata = build_dataloaders(config)
    model = ConvVAE(config.image_size, config.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Device={device}; train={len(train_loader.dataset)}; test={len(test_loader.dataset)}; " f"parameters={parameter_count:,}; size={config.image_size}")
    start = time.perf_counter()
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        current_beta = beta_for_epoch(epoch, config.beta, config.beta_warmup_epochs)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(model, train_loader, device, current_beta, config.l1_weight, config.bce_weight, config.edge_weight, optimizer=optimizer,)
        test_metrics = run_epoch(model, test_loader, device, current_beta, config.l1_weight, config.bce_weight, config.edge_weight,)
        entry = {
            "epoch": epoch,
            "beta": current_beta,
            "learning_rate": learning_rate,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(entry)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{config.epochs}: train={train_metrics['loss']:.3f} " f"test={test_metrics['loss']:.3f} recon={train_metrics['reconstruction']:.3f} " f"kl={train_metrics['kl']:.3f} beta={current_beta:.3f} " f"lr={learning_rate:.2e} ({entry['seconds']:.1f}s)", flush=True,)

    training_seconds = time.perf_counter() - start
    save_loss_plot(history, output_dir / "loss_curves.png")
    save_examples(model, test_loader, output_dir, device)

    metrics: dict[str, object] = {}
    if not config.skip_metrics:
        metrics = compute_ignite_metrics(model, test_loader, device, min(config.eval_samples, len(test_loader.dataset)), seed=config.seed,)
        evaluation_manifest = metrics.pop("evaluation_manifest")
        (output_dir / "evaluation_manifest.json").write_text(json.dumps(evaluation_manifest, indent=2) + "\n", encoding="utf-8")
        metrics["evaluation_manifest_file"] = "evaluation_manifest.json"
    metrics.update({ "device": str(device), "model_version": "residual-edge-vae-v3", "parameter_count": parameter_count, "training_seconds": training_seconds, "final_train_loss": history[-1]["train_loss"], "final_test_loss": history[-1]["test_loss"], "final_train_reconstruction": history[-1]["train_reconstruction"], "final_train_kl": history[-1]["train_kl"], "final_train_l1": history[-1]["train_l1"], "final_train_bce": history[-1]["train_bce"], "final_train_edge": history[-1]["train_edge"], })
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "dataset": dataset_metadata,
        "metrics": metrics,
    }
    torch.save(checkpoint, output_dir / "vae_checkpoint.pt")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output_dir / "dataset.json").write_text(json.dumps(dataset_metadata, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

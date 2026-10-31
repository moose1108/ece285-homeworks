"""Reproducible Torchvision Inception evaluation with Ignite aggregators.

This is a documented Torchvision-pool3 protocol, not numerically interchangeable
with TensorFlow FID, pytorch-fid, or Clean-FID.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import ignite
import numpy as np
import PIL
import scipy
import torch
import torchvision
from ignite.metrics import FID, InceptionScore
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models import Inception_V3_Weights, inception_v3


PROTOCOL_VERSION = "torchvision-pool3-imagenet-normalized-v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_evaluation_inputs(dataset_size: int, sample_count: int, latent_dim: int, seed: int) -> tuple[list[int], torch.Tensor]:
    """Use private CPU generators so training/plotting cannot alter evaluation."""
    if sample_count < 2 or sample_count > dataset_size:
        raise ValueError(f"sample_count must be in [2, {dataset_size}], got {sample_count}")
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    latent_generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=index_generator)[:sample_count].tolist()
    latents = torch.randn(sample_count, latent_dim, generator=latent_generator, dtype=torch.float32)
    return indices, latents


def normalize_inception_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Normalize [0,1] RGB before Torchvision's internal transform_input step."""
    mean = rgb.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = rgb.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (rgb - mean) / std


def prepare_inception_input(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Expected N x 1 x H x W grayscale tensors in [0,1].")
    rgb = F.interpolate(images.repeat(1, 3, 1, 1), size=(299, 299), mode="bilinear", align_corners=False, antialias=True,)
    return normalize_inception_rgb(rgb)


def _relative_real_files(dataset, indices: list[int]) -> list[str]:
    if not hasattr(dataset, "samples") or not hasattr(dataset, "root"):
        return [f"dataset-index:{index}" for index in indices]
    root = Path(dataset.root).resolve()
    return [Path(dataset.samples[index][0]).resolve().relative_to(root).as_posix() for index in indices]


@torch.no_grad()
def compute_ignite_metrics(model, test_loader: DataLoader, device: torch.device, sample_count: int, seed: int = 285,) -> dict[str, object]:
    """Evaluate fixed real indices and prior draws; reuse pool3 for IS logits."""
    indices, latents = fixed_evaluation_inputs(len(test_loader.dataset), sample_count, model.latent_dim, seed)
    loader = DataLoader(Subset(test_loader.dataset, indices), batch_size=test_loader.batch_size or 8, shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(seed + 2),)
    weights = Inception_V3_Weights.IMAGENET1K_V1
    inception = inception_v3(weights=weights, transform_input=True).to(device).eval()
    classifier = inception.fc
    inception.fc = nn.Identity()
    fid = FID(num_features=2048, feature_extractor=nn.Identity(), device="cpu")
    inception_score = InceptionScore(num_features=1000, feature_extractor=nn.Identity(), device="cpu")
    fid.reset()
    inception_score.reset()
    model.eval()

    processed = 0
    for real, _ in loader:
        count = real.shape[0]
        fake = model.decode(latents[processed : processed + count].to(device))
        real_features = inception(prepare_inception_input(real.to(device)))
        fake_features = inception(prepare_inception_input(fake))
        fake_probabilities = torch.softmax(classifier(fake_features), dim=1)
        fid.update((fake_features.cpu(), real_features.cpu()))
        inception_score.update(fake_probabilities.cpu())
        processed += count
        print(f"Metric features: {processed}/{sample_count}", flush=True)

    print("Computing Ignite FID covariance square root...", flush=True)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "real_dataset_root": "test",
        "real_dataset_size": len(test_loader.dataset),
        "real_indices": indices,
        "real_files": _relative_real_files(test_loader.dataset, indices),
        "real_index_seed": seed + 1,
        "latent_seed": seed,
        "latent_distribution": "N(0,I), torch.randn on CPU float32",
        "latent_shape": list(latents.shape),
        "latent_sha256": hashlib.sha256(latents.numpy().tobytes()).hexdigest(),
        "sample_count": sample_count,
        "batch_size": loader.batch_size,
        "image_preprocessing": "grayscale; aspect-preserving bilinear antialiased resize; center crop to VAE image size; [0,1]",
        "inception_preprocessing": "repeat grayscale to RGB; bilinear antialiased resize to 299x299; ImageNet mean/std normalization; Torchvision transform_input=True",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "weights": "torchvision.models.Inception_V3_Weights.IMAGENET1K_V1",
        "weights_url": weights.url,
        "fid_features": "2048-D average-pool features before fc",
        "is_features": "1000-class softmax from the same model's original fc",
        "aggregation": "ignite.metrics.FID and ignite.metrics.InceptionScore; CPU",
        "comparison_limit": "Not canonical TensorFlow/pytorch-fid/Clean-FID; compare only identical protocol and sample count.",
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pytorch_ignite": ignite.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": PIL.__version__,
        },
    }
    return {
        "fid": float(fid.compute()),
        "inception_score": float(inception_score.compute()),
        "metric_samples": processed,
        "feature_space": "Torchvision ImageNet Inception-v3 2048-D pool3 features",
        "is_space": "Torchvision ImageNet Inception-v3 1000-D probabilities",
        "resize": "bilinear antialiased to 299x299; grayscale repeated to RGB",
        "normalization": "ImageNet mean/std; Torchvision transform_input=True",
        "evaluation_protocol": PROTOCOL_VERSION,
        "evaluation_seed": seed,
        "evaluation_manifest": manifest,
    }

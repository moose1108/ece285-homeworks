"""Paired-image reconstruction metrics for controlled VAE diagnostics.

Inputs are grayscale tensors in [0, 1] with shape N x 1 x H x W. Every
reported quantity is calculated separately for each image and then averaged
over images, so batches of different sizes must be combined with sample-count
weights. PSNR is the average of image-wise PSNR values, not PSNR of batch MSE.

SSIM follows the single-scale Wang et al. formulation: a normalized 11 x 11
Gaussian window (sigma 1.5), valid convolution, population moments, data range
1, K1=0.01 and K2=0.03. There is no resizing or multiscale pooling here.
Reference: https://ece.uwaterloo.ca/~z70wang/research/ssim/

Finite-difference gradient magnitude/error are descriptive diagnostics only;
larger gradients do not establish sharper, anatomically correct images.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
PSNR_MSE_FLOOR = 1e-12


def _validate_pair(target: torch.Tensor, reconstruction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(target, torch.Tensor) or not isinstance(reconstruction, torch.Tensor):
        raise TypeError("Metrics require two torch.Tensor inputs.")
    if target.shape != reconstruction.shape:
        raise ValueError("Target and reconstruction must have identical shapes.")
    if target.ndim != 4 or target.shape[1] != 1 or target.shape[0] < 1:
        raise ValueError("Expected a nonempty N x 1 x H x W grayscale batch.")
    if min(target.shape[-2:]) < SSIM_WINDOW_SIZE:
        raise ValueError("SSIM requires image height and width of at least 11.")
    if target.device != reconstruction.device:
        raise ValueError("Target and reconstruction must be on the same device.")
    if not target.is_floating_point() or not reconstruction.is_floating_point():
        raise TypeError("Inputs must be floating-point image tensors in [0,1].")
    for name, images in (("target", target), ("reconstruction", reconstruction)):
        if not torch.isfinite(images).all().item():
            raise ValueError(f"{name} contains non-finite values.")
        if images.min().item() < 0 or images.max().item() > 1:
            raise ValueError(f"{name} must be in [0,1]; metrics never silently clamp pixels.")
    dtype = torch.float64 if torch.float64 in (target.dtype, reconstruction.dtype) else torch.float32
    return target.to(dtype=dtype), reconstruction.to(dtype=dtype)


def _ssim_per_image(target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    coordinates = torch.arange(SSIM_WINDOW_SIZE, dtype=target.dtype, device=target.device) - SSIM_WINDOW_SIZE // 2
    gaussian = torch.exp(-coordinates.square() / (2 * SSIM_SIGMA**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).view(1, 1, SSIM_WINDOW_SIZE, SSIM_WINDOW_SIZE)

    target_mean = F.conv2d(target, window)
    reconstruction_mean = F.conv2d(reconstruction, window)
    target_mean_squared = target_mean.square()
    reconstruction_mean_squared = reconstruction_mean.square()
    mean_product = target_mean * reconstruction_mean
    target_variance = F.conv2d(target.square(), window) - target_mean_squared
    reconstruction_variance = (F.conv2d(reconstruction.square(), window) - reconstruction_mean_squared)
    covariance = F.conv2d(target * reconstruction, window) - mean_product
    c1, c2 = 0.01**2, 0.03**2
    similarity = ((2 * mean_product + c1) * (2 * covariance + c2) / ( (target_mean_squared + reconstruction_mean_squared + c1) * (target_variance + reconstruction_variance + c2) ))
    # Keep floating-point roundoff from returning values just beyond the SSIM range.
    return similarity.flatten(1).mean(1).clamp(-1, 1)


@torch.no_grad()
def batch_reconstruction_metrics(target: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    """Return image-averaged MAE, MSE, PSNR(dB), SSIM, and gradient diagnostics.

    PSNR is capped at 120 dB via an MSE floor of 1e-12, including exact matches.
    Gradients are forward differences; each orientation is averaged over its
    valid pixels, then the horizontal/vertical means receive equal weight.
    """
    target, reconstruction = _validate_pair(target, reconstruction)
    error = reconstruction - target
    mae = error.abs().flatten(1).mean(1)
    mse = error.square().flatten(1).mean(1)
    psnr = -10 * torch.log10(mse.clamp_min(PSNR_MSE_FLOOR))
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    reconstruction_dx = reconstruction[:, :, :, 1:] - reconstruction[:, :, :, :-1]
    reconstruction_dy = reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :]
    target_gradient = 0.5 * (target_dx.abs().flatten(1).mean(1) + target_dy.abs().flatten(1).mean(1))
    reconstruction_gradient = 0.5 * (reconstruction_dx.abs().flatten(1).mean(1) + reconstruction_dy.abs().flatten(1).mean(1))
    gradient_mae = 0.5 * ((target_dx - reconstruction_dx).abs().flatten(1).mean(1) + (target_dy - reconstruction_dy).abs().flatten(1).mean(1))
    per_image = {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim": _ssim_per_image(target, reconstruction),
        "target_gradient_magnitude": target_gradient,
        "reconstruction_gradient_magnitude": reconstruction_gradient,
        "gradient_mae": gradient_mae,
    }
    return {name: float(values.mean().item()) for name, values in per_image.items()}

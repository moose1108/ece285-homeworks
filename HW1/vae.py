"""Residual variational autoencoder for square grayscale radiographs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    """Choose a GroupNorm group count that divides the channel count."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualDownBlock(nn.Module):
    """Residual block that halves spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1), nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(_groups(out_channels), out_channels),)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class ResidualUpBlock(nn.Module):
    """Residual resize-convolution block that doubles spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(_groups(out_channels), out_channels),)
        self.skip = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(in_channels, out_channels, 1),)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class ConvVAE(nn.Module):
    """Residual convolutional VAE supporting 64x64 and 128x128 inputs."""

    def __init__(self, image_size: int = 128, latent_dim: int = 128) -> None:
        super().__init__()
        if image_size not in (64, 128):
            raise ValueError("This architecture supports 64x64 or 128x128 inputs.")

        self.image_size = image_size
        self.latent_dim = latent_dim
        down_channels = [32, 64, 128, 256]
        if image_size == 128:
            down_channels.append(256)

        self.stem = nn.Conv2d(1, 32, 3, padding=1)
        encoder_blocks: list[nn.Module] = []
        in_channels = 32
        for out_channels in down_channels:
            encoder_blocks.append(ResidualDownBlock(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_blocks)

        self.bottleneck_channels = down_channels[-1]
        self.bottleneck_size = image_size // (2 ** len(down_channels))
        hidden_dim = self.bottleneck_channels * self.bottleneck_size**2
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        decoder_channels = list(reversed(down_channels[:-1])) + [16]
        decoder_blocks: list[nn.Module] = []
        in_channels = self.bottleneck_channels
        for out_channels in decoder_channels:
            decoder_blocks.append(ResidualUpBlock(in_channels, out_channels))
            in_channels = out_channels
        self.decoder = nn.Sequential(*decoder_blocks)
        self.output = nn.Sequential(nn.Conv2d(in_channels, 1, 3, padding=1), nn.Sigmoid())

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(self.stem(x)).flatten(start_dim=1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_input(z).view(-1, self.bottleneck_channels, self.bottleneck_size, self.bottleneck_size)
        return self.output(self.decoder(hidden))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    @torch.no_grad()
    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        return self.decode(torch.randn(count, self.latent_dim, device=device))


def vae_loss(reconstruction: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 0.5, l1_weight: float = 0.75, bce_weight: float = 0.05, edge_weight: float = 0.2,) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return total, reconstruction, KL, L1, BCE, and edge losses per image."""
    batch_size = target.shape[0]
    l1_loss = F.l1_loss(reconstruction, target, reduction="sum") / batch_size
    bce_loss = F.binary_cross_entropy(reconstruction, target, reduction="sum") / batch_size
    reconstruction_dx = reconstruction[:, :, :, 1:] - reconstruction[:, :, :, :-1]
    reconstruction_dy = reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    edge_loss = (F.l1_loss(reconstruction_dx, target_dx, reduction="sum") + F.l1_loss(reconstruction_dy, target_dy, reduction="sum")) / batch_size
    reconstruction_loss = (l1_weight * l1_loss + bce_weight * bce_loss + edge_weight * edge_loss)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / batch_size
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss, l1_loss, bce_loss, edge_loss

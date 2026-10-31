"""Pure Gaussian VAE with global and spatial standard-normal latent codes.

The spatial code is transformed by a learned Gaussian affine map conditioned
on the decoder's global context. The decoder input prior remains N(0,I).
Residual mode returns a diagonal-Gaussian posterior for that complete code.
Conditional-absolute mode learns absolute local posterior parameters, then
whitens them against the conditional decoder Gaussian at a sampled global
code. Its posterior factorization is q(g|x)q(eps|g,x), rather than a joint
diagonal Gaussian. Summed Gaussian KL is then global KL plus conditional
local KL, with a Monte Carlo sample for the expectation over q(g|x).

The local projection starts at zero, so its initial decoder output equals
the warm-start global VAE exactly. Both posterior heads remain learned.
Conditional decoder log-variance alone is clamped to [-8,8] for stability;
the posterior log-variance is not clipped or frozen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from vae import ConvVAE


PURE_BASELINE_SHA256 = "a59171e78e059a24e6be1dcaa03265f321471851a573cc54943ed874d5c7483c"


class SpatialVAE(nn.Module):
    """A global128-D code plus a4x16x16 spatial code for128px images."""

    def __init__(self, image_size: int = 128, global_dim: int = 128, local_channels: int = 4, local_size: int = 16, posterior_mode: str = "residual") -> None:
        super().__init__()
        if image_size != 128 or global_dim != 128 or local_size != 16:
            raise ValueError("This warm-start architecture requires image_size128, global_dim128 and local_size16")
        if isinstance(local_channels, bool) or not isinstance(local_channels, int) or local_channels < 1:
            raise ValueError("local_channels must be a positive integer")
        if posterior_mode not in ("residual", "conditional_absolute"):
            raise ValueError("posterior_mode must be residual or conditional_absolute")
        self.image_size = image_size
        self.global_dim = global_dim
        self.local_channels = local_channels
        self.local_size = local_size
        self.local_dim = local_channels * local_size * local_size
        self.latent_dim = global_dim + self.local_dim
        self.posterior_mode = posterior_mode
        self.base = ConvVAE(image_size=image_size, latent_dim=global_dim)
        self.local_posterior = nn.Conv2d(128, 2 * local_channels, kernel_size=3, padding=1)
        self.conditional_prior = nn.Conv2d(128, 2 * local_channels, kernel_size=3, padding=1)
        self.local_projection = nn.Conv2d(local_channels, 128, kernel_size=1)
        with torch.no_grad():
            self.local_posterior.weight.zero_()
            self.local_posterior.weight[:local_channels].normal_(mean=0.0, std=0.005)
            self.local_posterior.bias.zero_()
            self.local_posterior.bias[local_channels:].fill_(-2.0)
            nn.init.zeros_(self.conditional_prior.weight)
            nn.init.zeros_(self.conditional_prior.bias)
            nn.init.zeros_(self.local_projection.weight)
            nn.init.zeros_(self.local_projection.bias)
        posterior_description = "q(g|x)q(eps|g,x); absolute local q whitened using conditional decoder Gaussian at the same sampled global code" if posterior_mode == "conditional_absolute" else "Learned diagonal Gaussian for concatenated global and spatial standard-normal codes"
        self.model_config = {"architecture": "spatial_vae_v1", "image_size": image_size, "global_dim": global_dim, "local_channels": local_channels, "local_size": local_size, "latent_dim": self.latent_dim, "posterior_mode": posterior_mode, "conditional_decoder_logvar_bounds": [-8.0, 8.0], "posterior": posterior_description, "prior": "N(0,I) over all latent_dim coordinates"}

    def load_pure_base_checkpoint(self, checkpoint_path: str | Path) -> dict:
        """Load a verified original or recorded pure global-VAE warm start."""
        path = Path(checkpoint_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, map_location="cpu", weights_only=digest != PURE_BASELINE_SHA256)
        if not isinstance(checkpoint, dict):
            raise ValueError("The warm-start checkpoint must be a dictionary")
        recorded_pure = checkpoint.get("model_type") == "pure_vae" and checkpoint.get("initial_checkpoint_sha256") == PURE_BASELINE_SHA256
        if digest != PURE_BASELINE_SHA256 and not recorded_pure:
            raise ValueError("Warm start must belong to the preserved pure-VAE ancestry")
        if "discriminator_state_dict" in checkpoint or "discriminator_optimizer" in checkpoint:
            raise ValueError("Warm start contains incompatible additional network state")
        self.base.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.base.requires_grad_(True)
        return {"checkpoint_sha256": digest, "initial_checkpoint_sha256": PURE_BASELINE_SHA256, "source_model_type": "pure_vae", "source_step": checkpoint.get("step", 0), "projection_initialization": "Zero; decoder output initially equals the loaded global-only VAE"}

    def _encoder_parameters(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1:] != (1, self.image_size, self.image_size):
            raise ValueError("Expected N x1 x128 x128 image tensors")
        hidden = self.base.stem(images)
        local_features = None
        for index, block in enumerate(self.base.encoder):
            hidden = block(hidden)
            if index == 2:
                local_features = hidden
        if local_features is None or local_features.shape[1:] != (128, self.local_size, self.local_size):
            raise RuntimeError("The base encoder does not expose the required128x16x16 feature map")
        global_features = hidden.flatten(1)
        global_mu = self.base.fc_mu(global_features)
        global_logvar = self.base.fc_logvar(global_features)
        local_mu, local_logvar = self.local_posterior(local_features).chunk(2, dim=1)
        return global_mu, global_logvar, local_mu, local_logvar

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return monitor parameters, using global mean for conditional mode.

        Conditional training must call forward_with_noise: its local posterior
        parameters depend on the sampled global code, not its mean. This
        method ensures decode(mu) remains a correct deterministic monitor.
        """
        global_mu, global_logvar, local_mu, local_logvar = self._encoder_parameters(images)
        if self.posterior_mode == "conditional_absolute":
            context = self._decoder_context(global_mu)
            prior_mu, prior_logvar = self._conditional_parameters(context)
            local_mu = (local_mu - prior_mu) * torch.exp(-0.5 * prior_logvar)
            local_logvar = local_logvar - prior_logvar
        return torch.cat((global_mu, local_mu.flatten(1)), dim=1), torch.cat((global_logvar, local_logvar.flatten(1)), dim=1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def _conditional_parameters(self, global_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conditional_mu, conditional_logvar = self.conditional_prior(global_context).chunk(2, dim=1)
        return conditional_mu, conditional_logvar.clamp(-8.0, 8.0)

    def inject_local(self, global_context: torch.Tensor, local_code: torch.Tensor) -> torch.Tensor:
        """Transform a local standard-normal code inside the decoder."""
        conditional_mu, conditional_logvar = self._conditional_parameters(global_context)
        local_actual = conditional_mu + torch.exp(0.5 * conditional_logvar) * local_code
        return global_context + self.local_projection(local_actual)

    def _decoder_context(self, global_code: torch.Tensor) -> torch.Tensor:
        hidden = self.base.decoder_input(global_code).reshape(-1, self.base.bottleneck_channels, self.base.bottleneck_size, self.base.bottleneck_size)
        for block in self.base.decoder[:2]:
            hidden = block(hidden)
        return hidden

    def _finish_decode(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.base.decoder[2:]:
            hidden = block(hidden)
        return self.base.output(hidden)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected N x{self.latent_dim} standard-normal code tensors")
        global_code = latent[:, :self.global_dim]
        local_code = latent[:, self.global_dim:].reshape(-1, self.local_channels, self.local_size, self.local_size)
        context = self._decoder_context(global_code)
        return self._finish_decode(self.inject_local(context, local_code))

    def forward_with_noise(self, images: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample global code first and share its context with conditional KL."""
        if noise.ndim != 2 or noise.shape != (len(images), self.latent_dim):
            raise ValueError(f"Expected explicit posterior noise with shape N x{self.latent_dim}")
        if self.posterior_mode == "residual":
            mu, logvar = self.encode(images)
            return self.decode(mu + torch.exp(0.5 * logvar) * noise), mu, logvar
        global_mu, global_logvar, absolute_mu, absolute_logvar = self._encoder_parameters(images)
        global_code = global_mu + torch.exp(0.5 * global_logvar) * noise[:, :self.global_dim]
        context = self._decoder_context(global_code)
        prior_mu, prior_logvar = self._conditional_parameters(context)
        local_mu = (absolute_mu - prior_mu) * torch.exp(-0.5 * prior_logvar)
        local_logvar = absolute_logvar - prior_logvar
        local_noise = noise[:, self.global_dim:].reshape(-1, self.local_channels, self.local_size, self.local_size)
        local_epsilon = local_mu + torch.exp(0.5 * local_logvar) * local_noise
        local_actual = prior_mu + torch.exp(0.5 * prior_logvar) * local_epsilon
        reconstruction = self._finish_decode(context + self.local_projection(local_actual))
        mu = torch.cat((global_mu, local_mu.flatten(1)), dim=1)
        logvar = torch.cat((global_logvar, local_logvar.flatten(1)), dim=1)
        return reconstruction, mu, logvar

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.posterior_mode == "conditional_absolute":
            noise = torch.randn(len(images), self.latent_dim, device=images.device, dtype=images.dtype)
            return self.forward_with_noise(images, noise)
        mu, logvar = self.encode(images)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    @torch.no_grad()
    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        return self.decode(torch.randn(count, self.latent_dim, device=device))

"""Check the pure-VAE objective against Gaussian probability arithmetic."""

import unittest

import torch

from train_pure import pure_vae_loss


class PureTrainingTests(unittest.TestCase):
    def test_sum_per_image_and_analytic_kl_match_distribution_reference(self) -> None:
        target = torch.zeros(2, 1, 3, 3, dtype=torch.float64)
        reconstruction = torch.full_like(target, 0.25, requires_grad=True)
        mu = torch.ones(2, 4, dtype=torch.float64, requires_grad=True)
        logvar = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
        total, reconstruction_loss, kl = pure_vae_loss(reconstruction, target, mu, logvar, beta=0.1)
        posterior = torch.distributions.Normal(mu, (0.5 * logvar).exp())
        prior = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
        reference_kl = torch.distributions.kl_divergence(posterior, prior).sum(1).mean()
        torch.testing.assert_close(kl, reference_kl, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(float(reconstruction_loss.detach()), 2.25)
        self.assertAlmostEqual(float(total.detach()), 2.45)
        total.backward()
        self.assertGreater(float(reconstruction.grad.abs().sum()), 0)
        self.assertGreater(float(mu.grad.abs().sum()), 0)
        self.assertIsNotNone(logvar.grad)

    def test_batch_duplication_preserves_average_loss_and_posterior_noise_remains_learned(self) -> None:
        generator = torch.Generator().manual_seed(285)
        mu = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
        logvar = torch.full_like(mu, -1.0, requires_grad=True)
        noise = torch.randn(3, 4, generator=generator, dtype=torch.float64)
        reconstruction = (mu + (0.5 * logvar).exp() * noise).sigmoid().view(3, 1, 2, 2)
        target = torch.zeros_like(reconstruction)
        original = pure_vae_loss(reconstruction, target, mu, logvar, beta=0.1)
        duplicated = pure_vae_loss(reconstruction.repeat(2, 1, 1, 1), target.repeat(2, 1, 1, 1), mu.repeat(2, 1), logvar.repeat(2, 1), beta=0.1)
        for left, right in zip(original, duplicated, strict=True):
            torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
        original[0].backward()
        self.assertGreater(float(logvar.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(logvar.grad).all())


if __name__ == "__main__":
    unittest.main()

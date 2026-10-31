"""Check the optional structural reconstruction term without a learned critic."""

import unittest

import torch

from diagnostic_metrics import _ssim_per_image
from train_pure import pure_vae_loss, pure_vae_objective, ssim_reconstruction_loss


class PureSSIMTests(unittest.TestCase):
    def test_zero_weight_keeps_original_l1_kl_and_skips_structural_term(self) -> None:
        images = torch.zeros(2, 1, 3, 3, dtype=torch.float64)
        reconstruction = torch.ones_like(images) * 0.25
        mu, logvar = torch.ones(2, 4, dtype=torch.float64), torch.zeros(2, 4, dtype=torch.float64)
        base = pure_vae_loss(reconstruction, images, mu, logvar, beta=0.1)
        extended = pure_vae_objective(reconstruction, images, mu, logvar, beta=0.1, ssim_weight=0)
        self.assertIsNone(extended[3])
        for expected, actual in zip(base, extended[:3], strict=True):
            self.assertTrue(torch.equal(expected, actual))

    def test_structural_weight_is_scaled_by_pixels_once_and_keeps_image_gradients(self) -> None:
        generator = torch.Generator().manual_seed(285)
        images = torch.rand(2, 1, 16, 16, generator=generator, dtype=torch.float64)
        reconstruction = torch.rand(2, 1, 16, 16, generator=generator, dtype=torch.float64, requires_grad=True)
        mu = torch.ones(2, 4, dtype=torch.float64, requires_grad=True)
        logvar = torch.full_like(mu, -0.5, requires_grad=True)
        total, l1, kl, structural = pure_vae_objective(reconstruction, images, mu, logvar, beta=0.1, ssim_weight=0.2)
        reference = 1 - _ssim_per_image(images, reconstruction).mean()
        torch.testing.assert_close(structural, reference, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(total, l1 + 0.1 * kl + 0.2 * 256 * reference, atol=1e-12, rtol=1e-12)
        structural_gradient = torch.autograd.grad(structural, reconstruction, retain_graph=True)[0]
        self.assertGreater(float(structural_gradient.abs().sum()), 0)
        total.backward()
        for value in (reconstruction, mu, logvar):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_identical_images_have_zero_structural_error(self) -> None:
        images = torch.linspace(0, 1, 16 * 16, dtype=torch.float64).reshape(1, 1, 16, 16)
        self.assertAlmostEqual(float(ssim_reconstruction_loss(images, images)), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()

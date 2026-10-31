"""Analytic and edge-case checks for image reconstruction diagnostics."""

import math
import unittest

import torch

from diagnostic_metrics import batch_reconstruction_metrics


class DiagnosticMetricsTests(unittest.TestCase):
    def test_identical_images_have_perfect_similarity_and_finite_psnr(self):
        generator = torch.Generator().manual_seed(285)
        image = torch.rand(3, 1, 19, 23, generator=generator, dtype=torch.float64)
        metrics = batch_reconstruction_metrics(image, image)
        self.assertEqual(metrics["mae"], 0)
        self.assertEqual(metrics["mse"], 0)
        self.assertEqual(metrics["gradient_mae"], 0)
        self.assertEqual(metrics["psnr"], 120)
        self.assertAlmostEqual(metrics["ssim"], 1, places=12)
        self.assertEqual(metrics["target_gradient_magnitude"], metrics["reconstruction_gradient_magnitude"])
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_constant_images_match_closed_form_ssim_and_pixel_errors(self):
        target = torch.full((2, 1, 16, 17), 0.25, dtype=torch.float64)
        reconstruction = torch.full_like(target, 0.75)
        metrics = batch_reconstruction_metrics(target, reconstruction)
        expected_ssim = (2 * 0.25 * 0.75 + 0.01**2) / (0.25**2 + 0.75**2 + 0.01**2)
        self.assertAlmostEqual(metrics["ssim"], expected_ssim, places=10)
        self.assertEqual(metrics["mae"], 0.5)
        self.assertEqual(metrics["mse"], 0.25)
        self.assertAlmostEqual(metrics["psnr"], -10 * math.log10(0.25), places=12)
        self.assertEqual(metrics["gradient_mae"], 0)
        self.assertEqual(metrics["target_gradient_magnitude"], 0)

    def test_metrics_average_images_and_psnr_not_batch_mse(self):
        target = torch.zeros(2, 1, 16, 16, dtype=torch.float64)
        reconstruction = torch.stack([torch.full((1, 16, 16), 0.1), torch.full((1, 16, 16), 0.5)]).to(torch.float64)
        whole = batch_reconstruction_metrics(target, reconstruction)
        separately = [
            batch_reconstruction_metrics(target[index : index + 1], reconstruction[index : index + 1])
            for index in range(2)
        ]
        for key in whole:
            self.assertAlmostEqual(whole[key], sum(values[key] for values in separately) / 2, places=12)
        self.assertNotAlmostEqual(whole["psnr"], -10 * math.log10(whole["mse"]), places=2)

    def test_gradient_orientation_and_offset_invariance(self):
        ramp = torch.linspace(0.0, 0.8, 17, dtype=torch.float64).view(1, 1, 1, 17)
        target = ramp.repeat(1, 1, 13, 1)
        shifted = target + 0.1
        metrics = batch_reconstruction_metrics(target, shifted)
        # A horizontal step is 0.05; vertical differences are zero.
        self.assertAlmostEqual(metrics["target_gradient_magnitude"], 0.025, places=12)
        self.assertAlmostEqual(metrics["reconstruction_gradient_magnitude"], 0.025, places=12)
        self.assertAlmostEqual(metrics["gradient_mae"], 0, places=12)
        self.assertAlmostEqual(metrics["mae"], 0.1, places=12)

    def test_rejects_invalid_shape_range_and_nonfinite_pixels(self):
        good = torch.zeros(2, 1, 12, 12)
        bad_inputs = [
            torch.zeros(2, 3, 12, 12),
            torch.zeros(2, 1, 10, 12),
            torch.zeros(0, 1, 12, 12),
            torch.zeros(2, 12, 12),
        ]
        for bad in bad_inputs:
            with self.assertRaises(ValueError):
                batch_reconstruction_metrics(bad, bad)
        with self.assertRaises(ValueError):
            batch_reconstruction_metrics(good, good[:1])
        for value in (-0.001, 1.001, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                batch_reconstruction_metrics(good, torch.full_like(good, value))
        with self.assertRaises(TypeError):
            batch_reconstruction_metrics(good.to(torch.uint8), good.to(torch.uint8))


if __name__ == "__main__":
    unittest.main()

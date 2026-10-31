"""Regression checks for the corrected evaluator (no weights/download needed)."""

import unittest

import torch

from evaluation import (IMAGENET_MEAN, IMAGENET_STD, fixed_evaluation_inputs, normalize_inception_rgb, prepare_inception_input,)


class EvaluationTests(unittest.TestCase):
    def test_normalization_matches_torchvision_transform_input_contract(self):
        rgb = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1)
        normalized = normalize_inception_rgb(rgb)
        mean = rgb.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = rgb.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        # Torchvision transform_input converts ImageNet-normalized inputs to [-1,1].
        transformed = normalized * (std / 0.5) + (mean - 0.5) / 0.5
        torch.testing.assert_close(transformed, 2 * rgb - 1)

    def test_private_generators_ignore_global_random_state(self):
        indices, latents = fixed_evaluation_inputs(624, 512, 128, 285)
        torch.manual_seed(999)
        torch.randn(1000)
        other_indices, other_latents = fixed_evaluation_inputs(624, 512, 128, 285)
        self.assertEqual(indices, other_indices)
        torch.testing.assert_close(latents, other_latents, rtol=0, atol=0)
        self.assertEqual(len(set(indices)), 512)

    def test_batch_partition_preserves_fixed_latents(self):
        _, latents = fixed_evaluation_inputs(624, 32, 128, 285)
        torch.testing.assert_close(torch.cat(list(latents.split(8))), latents, rtol=0, atol=0)

    def test_shape_and_invalid_sample_counts(self):
        self.assertEqual(tuple(prepare_inception_input(torch.zeros(2, 1, 128, 128)).shape), (2, 3, 299, 299))
        for count in (0, 1, -1, 625):
            with self.assertRaises(ValueError):
                fixed_evaluation_inputs(624, count, 128, 285)


if __name__ == "__main__":
    unittest.main()

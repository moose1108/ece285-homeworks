"""Reject non-VAE training ancestry before loading an inference network."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import torch

from evaluate_pure import load_pure_model


BASELINE_SHA = "a59171e78e059a24e6be1dcaa03265f321471851a573cc54943ed874d5c7483c"


class PureEvaluationTests(unittest.TestCase):
    def test_adversarial_or_unverified_ancestry_is_rejected_before_model_loading(self) -> None:
        cases = ({"model_type": "vae_gan"}, {"model_type": "pure_vae", "initial_checkpoint_sha256": "unverified"}, {})
        for checkpoint in cases:
            with self.subTest(checkpoint=checkpoint), patch("evaluate_pure.torch.load", return_value=checkpoint), patch("evaluate_pure.sha256_file", return_value="different"), patch("evaluate_pure.ConvVAE") as constructor:
                with self.assertRaisesRegex(ValueError, "adversarial checkpoints are excluded"):
                    load_pure_model(Path("candidate.pt"), torch.device("cpu"))
                constructor.assert_not_called()

    def test_verified_pure_initializer_and_continuation_load_their_own_weights(self) -> None:
        cases = (({"config": {"image_size": 128, "latent_dim": 128}, "model_state_dict": {"weight": "fixture"}}, BASELINE_SHA), ({"model_type": "pure_vae", "initial_checkpoint_sha256": BASELINE_SHA, "config": {"image_size": 128, "latent_dim": 128}, "model_state_dict": {"weight": "fixture"}}, "new-pure-checkpoint-sha"))
        for checkpoint, digest in cases:
            model = MagicMock()
            model.to.return_value = model
            model.eval.return_value = model
            with self.subTest(checkpoint=checkpoint), patch("evaluate_pure.torch.load", return_value=checkpoint), patch("evaluate_pure.sha256_file", return_value=digest), patch("evaluate_pure.ConvVAE", return_value=model):
                loaded, metadata = load_pure_model(Path("candidate.pt"), torch.device("cpu"))
            self.assertIs(loaded, model)
            self.assertIs(metadata, checkpoint)
            model.load_state_dict.assert_called_once_with(checkpoint["model_state_dict"], strict=True)

    def test_spatial_checkpoint_uses_its_1152_dimension_architecture(self) -> None:
        checkpoint = {"model_type": "pure_vae", "initial_checkpoint_sha256": BASELINE_SHA, "config": {"architecture": "spatial", "image_size": 128, "global_latent_dim": 128, "local_channels": 4, "local_size": 16, "latent_dim": 1152}, "model_state_dict": {"weight": "fixture"}}
        model = MagicMock()
        model.latent_dim = 1152
        model.to.return_value = model
        model.eval.return_value = model
        with patch("evaluate_pure.torch.load", return_value=checkpoint), patch("spatial_vae.SpatialVAE", return_value=model) as constructor:
            loaded, _ = load_pure_model(Path("spatial.pt"), torch.device("cpu"))
        self.assertIs(loaded, model)
        constructor.assert_called_once_with(image_size=128, global_dim=128, local_channels=4, local_size=16, posterior_mode="residual")
        model.load_state_dict.assert_called_once_with(checkpoint["model_state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()

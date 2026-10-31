"""Small CPU checks for the pure global/spatial Gaussian VAE architecture."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from spatial_vae import SpatialVAE


class SpatialVAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    def test_zero_projection_preserves_base_decode_for_any_local_code(self) -> None:
        torch.manual_seed(285)
        model = SpatialVAE().eval()
        self.assertEqual(model.latent_dim, 1152)
        latent = torch.randn(1, model.latent_dim)
        alternate = latent.clone()
        alternate[:, model.global_dim:] = torch.randn_like(alternate[:, model.global_dim:]) * 3
        with torch.no_grad():
            expected = model.base.decode(latent[:, :model.global_dim])
            actual = model.decode(latent)
            other = model.decode(alternate)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(other, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (1, 1, 128, 128))

    def test_posterior_shapes_and_analytic_gaussian_kl_reach_local_head(self) -> None:
        torch.manual_seed(285)
        model = SpatialVAE()
        mu, logvar = model.encode(torch.rand(1, 1, 128, 128))
        self.assertEqual(mu.shape, (1, 1152))
        self.assertEqual(logvar.shape, mu.shape)
        torch.testing.assert_close(logvar[:, model.global_dim:], torch.full_like(logvar[:, model.global_dim:], -2.0), rtol=0, atol=0)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
        self.assertTrue(torch.isfinite(kl))
        kl.backward()
        for parameter in model.local_posterior.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_open_projection_reaches_local_code_and_conditional_decoder(self) -> None:
        torch.manual_seed(285)
        model = SpatialVAE()
        with torch.no_grad():
            model.local_projection.weight.fill_(0.01)
        context = torch.randn(2, 128, 16, 16, requires_grad=True)
        local_code = torch.randn(2, 4, 16, 16, requires_grad=True)
        output = model.inject_local(context, local_code)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(local_code.grad.abs().sum().item(), 0)
        for parameter in model.conditional_prior.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_rejects_incompatible_architecture_codes_and_warm_start(self) -> None:
        with self.assertRaises(ValueError):
            SpatialVAE(image_size=64)
        model = SpatialVAE()
        with self.assertRaises(ValueError):
            model.decode(torch.zeros(1, 128))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrong.pt"
            torch.save({"model_type": "other", "model_state_dict": {}}, path)
            with self.assertRaises(ValueError):
                model.load_pure_base_checkpoint(path)

    def test_conditional_absolute_kl_matches_gaussian_conditional_and_reconstructs_same_sample(self) -> None:
        torch.manual_seed(285)
        model = SpatialVAE(posterior_mode="conditional_absolute")
        with torch.no_grad():
            model.local_projection.weight.fill_(0.01)
            model.conditional_prior.weight.normal_(std=0.001)
            model.conditional_prior.bias[:model.local_channels].fill_(0.25)
            model.conditional_prior.bias[model.local_channels:].fill_(0.4)
        images = torch.rand(1, 1, 128, 128)
        noise = torch.randn(1, model.latent_dim)
        reconstruction, mu, logvar = model.forward_with_noise(images, noise)
        whitened_kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
        with torch.no_grad():
            global_mu, global_logvar, absolute_mu, absolute_logvar = model._encoder_parameters(images)
            global_code = global_mu + (0.5 * global_logvar).exp() * noise[:, :model.global_dim]
            context = model._decoder_context(global_code)
            prior_mu, prior_logvar = model._conditional_parameters(context)
            local_q = torch.distributions.Normal(absolute_mu, (0.5 * absolute_logvar).exp())
            local_p = torch.distributions.Normal(prior_mu, (0.5 * prior_logvar).exp())
            local_kl = torch.distributions.kl_divergence(local_q, local_p).flatten(1).sum(1).mean()
            global_q = torch.distributions.Normal(global_mu, (0.5 * global_logvar).exp())
            global_p = torch.distributions.Normal(torch.zeros_like(global_mu), torch.ones_like(global_mu))
            expected_kl = torch.distributions.kl_divergence(global_q, global_p).sum(1).mean() + local_kl
            absolute_sample = absolute_mu + (0.5 * absolute_logvar).exp() * noise[:, model.global_dim:].reshape_as(absolute_mu)
            expected_reconstruction = model._finish_decode(context + model.local_projection(absolute_sample))
        torch.testing.assert_close(whitened_kl, expected_kl, rtol=1e-5, atol=1e-3)
        torch.testing.assert_close(reconstruction, expected_reconstruction, rtol=1e-5, atol=1e-6)
        whitened_kl.backward()
        for parameter in model.conditional_prior.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_conditional_encode_mean_decodes_absolute_posterior_mean(self) -> None:
        torch.manual_seed(285)
        model = SpatialVAE(posterior_mode="conditional_absolute").eval()
        with torch.no_grad():
            model.local_projection.weight.fill_(0.01)
            model.conditional_prior.bias[:model.local_channels].fill_(0.4)
            model.conditional_prior.bias[model.local_channels:].fill_(0.7)
            images = torch.rand(1, 1, 128, 128)
            mu, _ = model.encode(images)
            actual = model.decode(mu)
            global_mu, _, absolute_mu, _ = model._encoder_parameters(images)
            context = model._decoder_context(global_mu)
            expected = model._finish_decode(context + model.local_projection(absolute_mu))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

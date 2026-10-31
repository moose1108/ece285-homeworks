# HW1 — Variational Autoencoder

This project trains a pure VAE to generate 128 × 128 grayscale chest X-ray images using version 2 of the Kaggle Chest X-Ray Images (Pneumonia) dataset. The selected conditional spatial VAE uses 128 global variables and 4 × 16 × 16 local variables, for a total of 1,152 standard-normal noise inputs during generation.

The model is trained with reconstruction loss and Gaussian KL divergence. It uses no GAN, discriminator, or adversarial loss. The Inception network is used only to calculate FID and IS.

## Files

| Path | Purpose |
| --- | --- |
| `vae.py`, `train.py` | Global Gaussian VAE and training from scratch |
| `spatial_vae.py`, `train_pure.py` | Selected conditional spatial VAE and continuation training |
| `evaluation.py`, `evaluate.py`, `evaluate_pure.py` | Fixed sampling, FID / IS, and visualization |
| `diagnostic_metrics.py`, `run_diagnostics.py` | Reconstruction metrics and diagnostic experiments |
| `download_data.py` | Download the pinned dataset version |
| `tests/` | 23 unit tests for the pure VAE and evaluation |
| `results/` | Metric summary, complete generated grids, comparisons, and reconstructions |
| `experiments/pure_vae_v1/`, `experiments/pure_vae_v2/`, `experiments/pure_vae_spatial_v2/` | Archived JSON configurations and records for each continuation stage |
| `experiments/diagnostics_v1/data_manifest.json` | Fixed image indices for the train-seen reconstruction monitor |
| `artifacts/` | Archived JSON configurations and evaluation records for the original baseline |
| `main.tex`, `report/figures/` | Report source and figures for Overleaf |
| `report/Homework_1_PureVAE.pdf` | Final two-page report |

Local workspace paths in the archived JSON files have been converted to relative paths. Original metrics and checkpoint SHA-256 hashes are preserved. The dataset, model weights, and optimizer states are not included in this repository.

## Environment and dataset

The experiments used Python 3.13. Package versions are pinned in `requirements.txt`. Run the following from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r HW1/requirements.txt
cd HW1
python download_data.py --version 2
```

The downloader creates `data/chest_xray`, containing `train/`, `test/`, and `val/`. If Kaggle requests authentication, sign in locally and keep credentials out of Git. The dataset contains 5,216 training images, 624 test images, and 16 validation images. The experiments did not use the official validation split.

Training selects CUDA, MPS, or CPU according to the available hardware. Torchvision downloads the Inception weights when FID / IS is calculated for the first time.

## Training and evaluation from scratch

The following commands train a new global VAE without requiring the archived checkpoints omitted from this repository. Outputs go to `runs/` to preserve the archived experiment records.

```bash
python train.py --output-dir runs/baseline --epochs 20 --seed 285 --skip-metrics
python evaluate.py --checkpoint runs/baseline/vae_checkpoint.pt --output-dir runs/baseline_evaluation --samples 512 --seed 285
```

For a short training smoke test, run:

```bash
python train.py --output-dir runs/smoke --epochs 1 --max-train-samples 32 --batch-size 8 --skip-metrics
```

A newly trained model is a separate experiment and does not automatically reproduce the archived final scores below.

## Selected model and archived weights

The selected model continues from the original 20-epoch baseline through two global VAE continuation stages, followed by conversion to a conditional spatial VAE and 3,000 training steps. The final stage uses a batch size of 16, AdamW with a learning rate decreasing from `1e-4` to `1e-5`, a KL weight of `beta=0.1`, and an SSIM weight ramping to `0.2` over 200 steps.

`train_pure.py` verifies the original checkpoint's SHA-256 hash. The `--resume` and `--warm-start` options also verify pure VAE ancestry. Reproducing the complete continuation path requires the original initializer and the appropriate warm-start / resume checkpoints separately. A new checkpoint trained with `train.py` has a different hash and cannot directly serve as the initializer for this archived experiment.

The following weights are **not included**, and this repository provides no download links for them:

| Weight path | SHA-256 |
| --- | --- |
| `artifacts/vae_checkpoint.pt` | `a59171e78e059a24e6be1dcaa03265f321471851a573cc54943ed874d5c7483c` |
| `experiments/pure_vae_spatial_v2/selected_model.pt` | `86444e6be01ed8cba6a484619388381cebc0bcc83b9c8c29e5e698177d56456b` |

After supplying the selected checkpoint separately, you can regenerate the complete sample grids for three fixed seeds and recalculate FID / IS. This example skips reconstruction evaluation:

```bash
python evaluate_pure.py --checkpoint experiments/pure_vae_spatial_v2/selected_model.pt --output-dir runs/selected_evaluation --samples 512 --seed 285 --grid-seeds 285 1285 2285 --skip-reconstruction
```

If you also supply the original baseline checkpoint, add `--baseline-checkpoint artifacts/vae_checkpoint.pt` to generate comparison grids. Remove `--skip-reconstruction` to recalculate train-seen reconstruction metrics using the split manifest preserved in this repository.

## Recorded results

| Model | FID ↓ | IS ↑ |
| --- | ---: | ---: |
| Original global VAE baseline | 376.3752 | 1.6890 |
| Selected conditional spatial VAE | 362.7236 | 1.8698 |

Both models use the same 512 test images, seed, and Inception evaluation procedure. Their latent dimensionalities differ, so the Gaussian latent vectors differ. The protocol is `torchvision-pool3-imagenet-normalized-v1`; these scores are not directly interchangeable with TensorFlow, pytorch-fid, or Clean-FID scores. This is a single-seed, 512-sample estimate, and the official test results had already been observed during earlier experiments.

Deterministic reconstruction for the selected model uses 255 images **already seen during training**: MAE `0.01724877`, SSIM `0.89084563`, and PSNR `32.6593 dB`. These are train-seen diagnostics, not held-out validation, and do not measure the sharpness of randomly generated images.

See [`results/metrics.json`](results/metrics.json) for full numerical results and sampling settings. Each generated grid contains all 25 samples at temperature 1, without individual image selection or sharpening. The images show chest and rib structure, with remaining blur and irregular textures.

![Seed 285: complete grid of 25 prior samples](results/generated_seed285.png)

- [Seed 1285](results/generated_seed1285.png)
- [Seed 2285](results/generated_seed2285.png)
- [Baseline / selected comparison](results/prior_comparison_seed285.png)
- [Train-seen reconstructions](results/reconstructions.png)
- [Final report PDF](report/Homework_1_PureVAE.pdf)

## Tests

Run from `HW1/`:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The tests cover Gaussian KL, SSIM, the spatial VAE, and evaluation ancestry checks. They do not require the dataset or archived weights.

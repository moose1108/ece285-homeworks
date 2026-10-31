"""Summarize completed controlled reconstruction experiments, without a report.

Default input: experiments/diagnostics_v1. Outputs a technical Markdown summary,
stage-separated scientific learning curves, and the actual reconstruction grids.
No experiments, checkpoints, training records, or report files are modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("overfit_l1_beta0", "overfit_hybrid_beta0", "vae_l1_beta05", "vae_hybrid_beta05", "vae_l1_beta005",)
PRIMARY_CONDITIONS = CONDITIONS[:4]
BETA_PAIR = ("vae_l1_beta05", "vae_l1_beta005")
GENERATION_CONDITIONS = ("vae_l1_beta05", "vae_hybrid_beta05", "vae_l1_beta005")
PAIRS = {
    "overfit": CONDITIONS[:2],
    "vae": CONDITIONS[2:4],
}
LABELS = {
    "overfit_l1_beta0": "AE / L1",
    "overfit_hybrid_beta0": "AE / hybrid",
    "vae_l1_beta05": "VAE / L1 + KL (beta 0.5)",
    "vae_hybrid_beta05": "VAE / hybrid + KL (beta 0.5)",
    "vae_l1_beta005": "VAE / L1 + KL (beta 0.05)",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiments(directory: Path, allow_partial: bool) -> tuple[dict, dict, list[str]]:
    experiments, histories = {}, {}
    missing = []
    for name in CONDITIONS:
        folder = directory / name
        metric_path = folder / "metrics.json"
        if not metric_path.exists():
            missing.append(name)
            continue
        metrics = read_json(metric_path)
        config = read_json(folder / "config.json")
        history = read_json(folder / "history.json")
        if metrics["configuration"] != config:
            raise ValueError(f"Configuration mismatch in {name}")
        if not history or history[0]["step"] != 0 or history[-1]["step"] != config["steps"]:
            raise ValueError(f"Incomplete final history in {name}")
        if metrics["initial_reconstruction"] != history[0]["evaluation"]:
            raise ValueError(f"Initial metrics/history mismatch in {name}")
        if metrics["final_reconstruction"] != history[-1]["evaluation"]:
            raise ValueError(f"Final metrics/history mismatch in {name}")
        required = ("mae", "mse", "ssim", "psnr", "gradient_mae")
        for phase in ("initial_reconstruction", "final_reconstruction"):
            if not all(math.isfinite(float(metrics[phase][key])) for key in required):
                raise ValueError(f"Non-finite reconstruction metrics in {name}/{phase}")
        experiments[name], histories[name] = metrics, history
    if missing and not allow_partial:
        raise FileNotFoundError("Final metrics are not available for: " + ", ".join(missing) + ". Wait for training to complete or use --allow-partial.")
    if not experiments:
        raise FileNotFoundError("No completed conditions found; no summary was written.")
    return experiments, histories, missing


def validate_controls(experiments: dict) -> dict:
    controls = {}
    equal_fields = ("initial_state_sha256", "batch_schedule_sha256", "posterior_noise_sha256", "training_images", "evaluation_images", "steps", "batch_size", "initial_learning_rate", "minimum_learning_rate", "optimizer", "target_beta", "warmup_steps", "deterministic_training_decode_mu", "deterministic_evaluation_decode_mu", "seed", "device",)
    for stage, names in PAIRS.items():
        if not all(name in experiments for name in names):
            controls[stage] = None
            continue
        left, right = [experiments[name]["configuration"] for name in names]
        for field in equal_fields:
            if left[field] != right[field]:
                raise ValueError(f"Loss comparison is not controlled: {stage}/{field}")
        controls[stage] = {
            "initialization": left["initial_state_sha256"],
            "batch_schedule": left["batch_schedule_sha256"],
            "posterior_noise": left["posterior_noise_sha256"],
        }
    if all(name in experiments for name in BETA_PAIR):
        left, right = [experiments[name]["configuration"] for name in BETA_PAIR]
        if left["target_beta"] != 0.5 or right["target_beta"] != 0.05:
            raise ValueError("Expected target beta 0.5 versus 0.05 in the beta comparison.")
        if left["reconstruction_weights"] != right["reconstruction_weights"]:
            raise ValueError("The beta comparison changed reconstruction weights.")
        if left["reconstruction_weights"] != {"l1": 1.0, "bce": 0.0, "edge": 0.0}:
            raise ValueError("The beta comparison must use L1-only reconstruction.")
        for field in equal_fields:
            if field != "target_beta" and left[field] != right[field]:
                raise ValueError(f"Beta comparison is not controlled: {field}")
        controls["beta"] = {
            "initialization": left["initial_state_sha256"],
            "batch_schedule": left["batch_schedule_sha256"],
            "posterior_noise": left["posterior_noise_sha256"],
        }
    else:
        controls["beta"] = None
    return controls


def settings_line(config: dict) -> str:
    return (f"{config['steps']} optimizer steps; batch {config['batch_size']}; " f"LR {config['initial_learning_rate']:g} to {config['minimum_learning_rate']:g}; " f"target beta {config['target_beta']:g}; warm-up {config['warmup_steps']} steps; " f"seed {config['seed']}; device {config['device']}.")


def metric_table(experiments: dict, names: tuple[str, ...]) -> list[str]:
    lines = [
        "| Condition | MAE initial -> final | SSIM initial -> final | PSNR initial -> final (dB) | Gradient MAE initial -> final |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        if name not in experiments:
            continue
        result = experiments[name]
        initial, final = result["initial_reconstruction"], result["final_reconstruction"]
        cells = [LABELS[name]]
        for key, digits in (("mae", 5), ("ssim", 4), ("psnr", 2), ("gradient_mae", 5)):
            cells.append(f"{initial[key]:.{digits}f} -> {final[key]:.{digits}f}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def weighted_term_table(experiments: dict, histories: dict) -> list[str]:
    lines = [
        "| Condition | Weighted L1 | Weighted BCE | Weighted edge | Weighted KL | Total objective |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CONDITIONS:
        if name not in experiments:
            continue
        result = experiments[name]
        config = result["configuration"]
        row = histories[name][-1]
        training = row["training_interval"]
        weights = config["reconstruction_weights"]
        terms = [training[key] * weights[key] for key in ("l1", "bce", "edge")]
        terms.append(training["kl"] * row["beta"])
        lines.append("| " + LABELS[name] + " | " + " | ".join(f"{value:.4f}" for value in (*terms, training["objective"])) + " |")
    return lines


def old_fid_table(root: Path) -> list[str]:
    candidates = (("Previous center-crop / bilinear model", root / "experiments/previous_center_corrected/metrics.json"), ("Existing final edge / nearest model", root / "artifacts/metrics.json"),)
    found = [(label, read_json(path)) for label, path in candidates if path.exists()]
    if not found:
        return []
    lines = [
        "## Existing full-training models: corrected evaluation",
        "",
        "This is a separate evaluation of the earlier full-training checkpoints, not the small-data diagnostic models above. "
        "The earlier models used 5,216 training images and 20 epochs; they are reference results rather than a matched-budget comparison "
        "against the diagnostic models trained on 1,024 images for 800 steps.",
        "",
        "| Existing checkpoint | Corrected FID | Corrected IS | Samples | Protocol |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for label, metrics in found:
        lines.append(f"| {label} | {metrics['fid']:.4f} | {metrics['inception_score']:.4f} " f"| {metrics['metric_samples']} | `{metrics.get('evaluation_protocol', 'not recorded')}` |")
    if len(found) == 2:
        comparable = (found[0][1].get("evaluation_protocol") == found[1][1].get("evaluation_protocol") and found[0][1]["metric_samples"] == found[1][1]["metric_samples"] and found[0][1].get("evaluation_seed") == found[1][1].get("evaluation_seed"))
        lines.extend([ "", "Protocol, sample-count, and evaluation-seed identifiers " + ("match." if comparable else "do not all match; do not rank these scores directly."), ])
    lines.extend([ "", "These Torchvision/Inception + Ignite values are not canonical TensorFlow-FID or Clean-FID. " "They use ImageNet features and one finite real/generated sample set. " "No FID is reported for deterministic beta=0 memorization diagnostics.", ])
    return lines


def generation_quality_table(directory: Path, experiments: dict) -> list[str]:
    """Read optional prior evaluations and validate their fixed evaluation inputs."""
    found, missing = {}, []
    for name in GENERATION_CONDITIONS:
        folder = directory / name
        metric_path, manifest_path = folder / "generation_metrics.json", folder / "generation_manifest.json"
        if not metric_path.exists() or not manifest_path.exists():
            missing.append(name)
            continue
        if name not in experiments:
            raise ValueError(f"Generation evaluation lacks completed reconstruction results: {name}")
        metrics, manifest = read_json(metric_path), read_json(manifest_path)
        config = experiments[name]["configuration"]
        if config["stage"] != "vae" or config["deterministic_training_decode_mu"] is not False:
            raise ValueError(f"Normal-prior evaluation is not valid for a deterministic AE: {name}")
        if metrics["condition"] != name or manifest["source_condition"] != name:
            raise ValueError(f"Generation condition metadata mismatch in {name}")
        if metrics["training_images"] != config["training_images"] or metrics["training_steps"] != config["steps"]:
            raise ValueError(f"Generation metrics have a different training budget in {name}")
        if metrics["target_beta"] != config["target_beta"] or metrics["reconstruction_weights"] != config["reconstruction_weights"]:
            raise ValueError(f"Generation metrics have a different training objective in {name}")
        if manifest["training_config"] != config:
            raise ValueError(f"Generation manifest/configuration mismatch in {name}")
        if metrics["metric_samples"] != 512 or manifest["sample_count"] != 512:
            raise ValueError(f"Expected 512 real/generated samples for the diagnostic generation comparison: {name}")
        if metrics["evaluation_protocol"] != "torchvision-pool3-imagenet-normalized-v1":
            raise ValueError(f"Corrected Inception normalization protocol is missing in {name}")
        if metrics["evaluation_protocol"] != manifest["protocol_version"]:
            raise ValueError(f"Generation protocol mismatch in {name}")
        if len(manifest["real_indices"]) != 512 or len(set(manifest["real_indices"])) != 512:
            raise ValueError(f"Real-image index list is not 512 unique images in {name}")
        if len(manifest["real_files"]) != 512 or len(set(manifest["real_files"])) != 512:
            raise ValueError(f"Real-image file list is not 512 unique images in {name}")
        if metrics["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
            raise ValueError(f"Generation checkpoint identifiers disagree in {name}")
        checkpoint = folder / "checkpoint.pt"
        if checkpoint.exists():
            digest = hashlib.sha256()
            with checkpoint.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != manifest["checkpoint_sha256"]:
                raise ValueError(f"Generation checkpoint content changed in {name}")
        if not all(math.isfinite(float(metrics[key])) for key in ("fid", "inception_score")):
            raise ValueError(f"Non-finite generation metric in {name}")
        found[name] = (metrics, manifest, config)

    lines = ["## Prior-generation quality of the matched VAE conditions", ""]
    if not found:
        lines.append("Generation evaluation is pending for all three stochastic VAE conditions; reconstruction metrics alone do not establish normal-prior sample quality.")
        return lines

    reference = next(iter(found.values()))
    input_fields = ("protocol_version", "sample_count", "real_indices", "real_files", "latent_sha256", "latent_shape", "latent_seed", "weights", "normalization_mean", "normalization_std", "inception_preprocessing",)
    budget_fields = ("training_images", "steps", "batch_size", "initial_learning_rate", "minimum_learning_rate", "optimizer", "warmup_steps", "seed", "initial_state_sha256", "batch_schedule_sha256", "posterior_noise_sha256",)
    for name, (_, manifest, config) in found.items():
        for field in input_fields:
            if manifest[field] != reference[1][field]:
                raise ValueError(f"Prior-generation evaluation inputs differ: {name}/{field}")
        for field in budget_fields:
            if config[field] != reference[2][field]:
                raise ValueError(f"Prior-generation training controls differ: {name}/{field}")
    lines.extend([ f"The three stochastic VAE conditions use the same {reference[2]['training_images']:,} training images, " f"{reference[2]['steps']} optimizer steps, " "initial state, minibatch order, posterior-noise draws, and optimizer settings. " "Their loss weights or target beta differ as documented above. The table places deterministic " "internal-validation reconstruction metrics alongside separate official-test/prior-generation evaluation.", "", "| Condition | Internal-validation MAE | Internal-validation SSIM | Corrected FID | Corrected IS | Real / generated samples |", "| --- | ---: | ---: | ---: | ---: | ---: |", ])
    for name in GENERATION_CONDITIONS:
        if name not in found:
            continue
        metrics, _, _ = found[name]
        reconstruction = experiments[name]["final_reconstruction"]
        lines.append(f"| {LABELS[name]} | {reconstruction['mae']:.5f} | {reconstruction['ssim']:.4f} " f"| {metrics['fid']:.4f} | {metrics['inception_score']:.4f} | 512 / 512 |")
    if missing:
        lines.extend(["", "Pending generation evaluations: " + ", ".join(f"`{name}`" for name in missing) + ". This table is incomplete."])
    else:
        lines.extend(["", "All three generation evaluations use matching 512-image real indices/files, latent SHA-256, normalization, weights, and protocol identifiers."])
    lines.extend([ "", "This is one training seed and one finite 512-sample evaluation per condition; there is no confidence interval " "or claim of statistical significance. The official test images were seen in earlier exploratory development, " "so they are not an untouched final holdout. FID/IS describe the documented Torchvision + Ignite feature protocol, " "not anatomical correctness or interchangeable TensorFlow/Clean-FID benchmarks. " "No normal-prior FID/IS is assigned to the deterministic beta-zero AE diagnostics.", ])
    return lines


def write_summary(directory: Path, experiments: dict, histories: dict, controls: dict, manifest: dict, missing: list[str], generation_lines: list[str] | None = None) -> Path:
    splits = manifest["splits"]
    lines = [
        "# Controlled reconstruction diagnostics",
        "",
        "Status: " + ("INCOMPLETE; completed conditions only." if missing else "All five conditions completed."),
        "",
        "The experiments retain the existing 128x128 residual VAE architecture and use deterministic `decode(mu)` "
        "for reconstruction measurement. No augmentation is applied. All diagnostic data comes from the official "
        "training split; the official test split is not used to select these experiments.",
        "",
        f"Internal data: {splits['train']['count']} training-subset images; {splits['validation']['count']} "
        f"validation images; {splits['overfit']['count']} images for the memorization diagnostic.",
        "Filename-inferred group separation is conservative and is **not verified patient metadata**. "
        "There is no inferred-group overlap between the training subset and internal validation.",
    ]
    if missing:
        lines.extend(["", "Incomplete conditions: " + ", ".join(f"`{name}`" for name in missing) + "."])
    for stage, title, description in (("overfit", "Deterministic autoencoder memorization", "These metrics are measured on the same 64 images used for training, with beta=0 and no posterior noise. They test reconstruction capacity, not generalization or normal-prior image generation. The untrained log-variance head is not interpreted."), ("vae", "VAE internal validation", "These metrics are measured on filename-group-separated internal validation images, using posterior means. Training uses matched Gaussian posterior noise and KL warm-up. These are small-data diagnostic results, not full-dataset model claims."),):
        names = PAIRS[stage]
        present = [name for name in names if name in experiments]
        if not present:
            continue
        lines.extend(["", "## " + title, "", description, "", settings_line(experiments[present[0]]["configuration"]), ""])
        lines.extend(metric_table(experiments, names))
        lines.extend(["", "Do not compare the absolute values in this stage against the other stage: the image sets and training tasks differ."])

    beta_present = [name for name in BETA_PAIR if name in experiments]
    if beta_present:
        config = experiments[beta_present[0]]["configuration"]
        lines.extend([ "", "## Controlled beta comparison on internal validation", "", "The additional L1-only VAE condition changes target beta from 0.5 to 0.05. " "Both runs retain the same images, initialization, optimizer, step budget, minibatch order, " "posterior noise, reconstruction weights, and warm-up length. Beta increases linearly over " f"{config['warmup_steps']} steps in each run, to the respective target. This comparison " "addresses the KL tradeoff directly; it does not compare memorized training images with held-out images.", "", ])
        lines.extend(metric_table(experiments, BETA_PAIR))
        lines.extend([ "", "The beta intervention is controlled for this single seed and fixed budget, but neither " "the direction nor the best beta is assumed in advance. A reconstruction improvement alone does " "not establish improved samples drawn from the normal prior. See the dedicated beta figures.", ])

    lines.extend(["", "## Matched controls", ""])
    for stage, fields in controls.items():
        if fields is None:
            lines.append(f"- {stage}: not yet verifiable because a paired condition is missing.")
            continue
        lines.append(f"- {stage}: initialization, minibatch schedule, optimizer settings, and training budget match within the paired comparison.")
        lines.append(f"  - Initialization SHA-256: `{fields['initialization']}`.")
        lines.append(f"  - Batch-schedule SHA-256: `{fields['batch_schedule']}`.")
        lines.append(f"  - Posterior-noise SHA-256: `{fields['posterior_noise']}`." if fields["posterior_noise"] is not None else "  - Posterior noise: not used (deterministic autoencoder diagnostic).")

    lines.extend([ "", "## Final logged training-interval terms", "", "The following are averages over the final logged training interval, not deterministic reconstruction " "metrics on validation. Raw L1/BCE/edge terms are multiplied by the configured weights; " "raw KL is multiplied by the final interval beta. Scalar magnitudes do not equal gradient influence.", "", ])
    lines.extend(weighted_term_table(experiments, histories))
    lines.extend([ "", "## Interpretation limits", "", "- A single seed and one internal split are exploratory evidence; there are no repeated-training confidence intervals.", "- L1-only and hybrid reconstruction objectives have different numerical scales. Keeping beta fixed changes " "the effective reconstruction/KL balance; that loss comparison is not an isolated edge-loss ablation. " "The separate L1-only beta comparison holds reconstruction weights fixed.", "- The AE and VAE stages differ in data size, exposure per image, evaluation data, stochasticity, and KL. " "Their reconstruction gap cannot be attributed solely to KL; the controlled beta pair addresses that confound.", "- Lower total objective across different loss definitions cannot establish superior image quality.", "- SSIM uses an 11x11 Gaussian window (sigma 1.5), valid convolution, and data range 1. " "PSNR is averaged per image with an MSE floor of 1e-12 (maximum 120 dB).", "- Gradient magnitudes and gradient MAE describe finite differences; more gradient energy does not establish sharper or anatomically correct images.", "- Check `reconstruction_comparison.png` alongside metrics. Each panel shows originals in its top row and reconstructions in its bottom row.", ])
    lines.extend(["", *(generation_lines if generation_lines is not None else generation_quality_table(directory, experiments))])
    lines.extend(["", *old_fid_table(ROOT)])
    path = directory / "SUMMARY.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def plot_curves(directory: Path, experiments: dict, histories: dict, manifest: dict, missing: list[str]) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    for row, stage in enumerate(("overfit", "vae")):
        names = PAIRS[stage]
        count = manifest["splits"]["overfit" if stage == "overfit" else "validation"]["count"]
        stage_label = f"AE: same {count} training images" if stage == "overfit" else f"VAE: {count} internal-validation images"
        for col, (key, ylabel) in enumerate((("mae", "MAE / pixel"), ("ssim", "SSIM"))):
            axis = axes[row, col]
            for name in names:
                if name not in experiments:
                    continue
                history = histories[name]
                axis.plot([entry["step"] for entry in history], [entry["evaluation"][key] for entry in history], marker="o", markersize=3, label=LABELS[name],)
            axis.set(title=stage_label, xlabel="Optimizer step", ylabel=ylabel)
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
            if any(name in experiments for name in names):
                axis.legend(frameon=False, fontsize=8)
            else:
                axis.text(0.5, 0.5, "No completed conditions", transform=axis.transAxes, ha="center")
    figure.suptitle("Deterministic reconstruction diagnostics" + (" - INCOMPLETE" if missing else ""), fontsize=13,)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output = directory / "controlled_comparison.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_reconstructions(directory: Path, experiments: dict, missing: list[str]) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(16, 7.2))
    for axis, name in zip(axes.flat, PRIMARY_CONDITIONS, strict=True):
        axis.axis("off")
        if name not in experiments:
            axis.set_title(LABELS[name] + " - incomplete")
            continue
        image_path = directory / name / "reconstructions.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        metrics = experiments[name]["final_reconstruction"]
        axis.imshow(mpimg.imread(image_path), cmap="gray", vmin=0, vmax=1)
        stage = "memorized training set" if name.startswith("overfit_") else "internal validation"
        axis.set_title(f"{LABELS[name]} ({stage})\nMAE {metrics['mae']:.5f}; SSIM {metrics['ssim']:.4f}; PSNR {metrics['psnr']:.2f} dB", fontsize=11,)
    figure.suptitle("Within each panel: originals above, deterministic reconstructions below" + (" - INCOMPLETE" if missing else ""), fontsize=13,)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output = directory / "reconstruction_comparison.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_beta_curves(directory: Path, experiments: dict, histories: dict, manifest: dict, missing: list[str]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    count = manifest["splits"]["validation"]["count"]
    for axis, (key, ylabel) in zip(axes, (("mae", "MAE / pixel"), ("ssim", "SSIM")), strict=True):
        for name in BETA_PAIR:
            if name not in experiments:
                continue
            history = histories[name]
            axis.plot([entry["step"] for entry in history], [entry["evaluation"][key] for entry in history], marker="o", markersize=3, label=LABELS[name],)
        axis.set(xlabel="Optimizer step", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        if any(name in experiments for name in BETA_PAIR):
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"Controlled beta comparison: L1 VAE, {count} internal-validation images" + (" - INCOMPLETE" if any(name in missing for name in BETA_PAIR) else ""), fontsize=12,)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output = directory / "beta_comparison.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_beta_reconstructions(directory: Path, experiments: dict, missing: list[str]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(16, 3.8))
    for axis, name in zip(axes, BETA_PAIR, strict=True):
        axis.axis("off")
        if name not in experiments:
            axis.set_title(LABELS[name] + " - incomplete")
            continue
        image_path = directory / name / "reconstructions.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        metrics = experiments[name]["final_reconstruction"]
        axis.imshow(mpimg.imread(image_path), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"{LABELS[name]} (same internal-validation images)\n" f"MAE {metrics['mae']:.5f}; SSIM {metrics['ssim']:.4f}; PSNR {metrics['psnr']:.2f} dB", fontsize=11,)
    figure.suptitle("Within each panel: originals above, deterministic reconstructions below" + (" - INCOMPLETE" if any(name in missing for name in BETA_PAIR) else ""), fontsize=13,)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    output = directory / "beta_reconstruction_comparison.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def build(directory: Path, allow_partial: bool = False) -> list[Path]:
    manifest = read_json(directory / "data_manifest.json")
    train_groups = set(manifest["splits"]["train"]["filename_groups"])
    validation_groups = set(manifest["splits"]["validation"]["filename_groups"])
    if train_groups & validation_groups:
        raise ValueError("Data manifest has inferred-group leakage.")
    experiments, histories, missing = load_experiments(directory, allow_partial)
    controls = validate_controls(experiments)
    generation_lines = generation_quality_table(directory, experiments)
    # Validate and plot first so missing image artifacts cannot yield a final-looking summary.
    outputs = [
        plot_curves(directory, experiments, histories, manifest, missing),
        plot_reconstructions(directory, experiments, missing),
        plot_beta_curves(directory, experiments, histories, manifest, missing),
        plot_beta_reconstructions(directory, experiments, missing),
        write_summary(directory, experiments, histories, controls, manifest, missing, generation_lines),
    ]
    for output in outputs:
        print(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "experiments/diagnostics_v1")
    parser.add_argument("--allow-partial", action="store_true", help="Label missing conditions as incomplete instead of failing")
    args = parser.parse_args()
    build(args.directory.resolve(), allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()

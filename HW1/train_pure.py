"""Continue the Gaussian VAE with reconstruction losses and Gaussian KL.

Training uses the official training split only. Both Gaussian heads remain
learned, and generation uses independent standard-normal latent vectors.
The reconstruction monitor is train-seen, including through the initializer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch-cache"))

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from diagnostic_metrics import _ssim_per_image, batch_reconstruction_metrics
from evaluation import sha256_file
from train import choose_device, make_transform
from vae import ConvVAE


INITIAL_SHA256 = "a59171e78e059a24e6be1dcaa03265f321471851a573cc54943ed874d5c7483c"
DISPLAY_SEEDS = (285, 1285, 2285)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def atomic_save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def pure_vae_loss(reconstruction: torch.Tensor, images: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image summed L1 plus beta times analytic diagonal-Gaussian KL."""
    l1 = F.l1_loss(reconstruction, images, reduction="sum") / len(images)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return l1 + beta * kl, l1, kl


def ssim_reconstruction_loss(reconstruction: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """Differentiable mean1-SSIM using the existing11px Gaussian-window metric."""
    return 1 - _ssim_per_image(images, reconstruction).mean()


def pure_vae_objective(reconstruction: torch.Tensor, images: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float, ssim_weight: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Keep zero-weight behavior exact; otherwise add pixels*weight*(1-SSIM)."""
    total, l1, kl = pure_vae_loss(reconstruction, images, mu, logvar, beta)
    structural = ssim_reconstruction_loss(reconstruction, images) if ssim_weight else None
    if structural is not None:
        total = total + ssim_weight * images[0].numel() * structural
    return total, l1, kl, structural


@torch.no_grad()
def save_monitor(model: ConvVAE, images: torch.Tensor, output: Path, step: int, batch_size: int) -> dict:
    model.eval()
    totals = {}
    all_mu, all_logvar, reconstructions = [], [], []
    for batch in images.split(batch_size):
        mu, logvar = model.encode(batch)
        decoded = model.decode(mu)
        for name, value in batch_reconstruction_metrics(batch, decoded).items():
            totals[name] = totals.get(name, 0.0) + len(batch) * value
        all_mu.append(mu.cpu())
        all_logvar.append(logvar.cpu())
        reconstructions.append(decoded.cpu())
    mu, logvar = torch.cat(all_mu), torch.cat(all_logvar)
    second = mu.square() + logvar.exp()
    decoded = torch.cat(reconstructions)
    save_image(torch.cat((images[:8].cpu(), decoded[:8])), output / "reconstructions.png", nrow=min(8, len(images)), padding=2)
    for seed in DISPLAY_SEEDS:
        latent = torch.randn(25, model.latent_dim, generator=torch.Generator(device="cpu").manual_seed(seed))
        generated = torch.cat([model.decode(batch.to(images.device)).cpu() for batch in latent.split(batch_size)])
        save_image(generated, output / f"generated_seed{seed}.png", nrow=5, padding=2)
        save_image(generated, output / f"prior_step_{step:06d}_seed{seed}.png", nrow=5, padding=2)
        if seed == 285:
            save_image(generated, output / "generated_samples.png", nrow=5, padding=2)
    return {"train_seen_reconstruction": {name: value / len(images) for name, value in totals.items()}, "raw_kl": float((-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)).mean()), "posterior_variance": float(logvar.exp().mean()), "aggregate_mean_abs": float(mu.mean(0).abs().mean()), "aggregate_radius_per_dimension": float(second.mean()), "aggregate_std_mean": float((second.mean(0) - mu.mean(0).square()).clamp_min(0).sqrt().mean()), "scope": "Deterministic decode(mu) on train-seen images; not held-out validation", "prior_sampling": "Three complete 25-image grids, private CPU N(0,I), temperature1, seeds285/1285/2285"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, default=ROOT / "artifacts/vae_checkpoint.pt")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path, help="Verified pure-VAE weights for an explicit architecture transition; optimizer restarts")
    parser.add_argument("--architecture", choices=("conv", "spatial"), default="conv")
    parser.add_argument("--local-channels", type=int, default=4)
    parser.add_argument("--spatial-posterior-mode", choices=("residual", "conditional_absolute"), default="residual")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/chest_xray")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/pure_vae_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--stop-at-step", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta-start", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--beta-ramp", type=int, default=200)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--ssim-ramp", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--max-train-images", type=int, help="Small CPU smoke tests only")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.log_every) < 1 or min(args.beta_ramp, args.ssim_ramp) < 0:
        parser.error("Steps, batch size and reporting intervals must be positive; ramps must be nonnegative")
    if not all(math.isfinite(value) and value >= 0 for value in (args.learning_rate, args.min_learning_rate, args.weight_decay, args.beta_start, args.beta, args.ssim_weight)) or args.learning_rate <= 0 or args.min_learning_rate > args.learning_rate:
        parser.error("Rates and loss settings must be finite, nonnegative and ordered")
    if args.max_train_images is not None and args.max_train_images < 1:
        parser.error("max-train-images must be positive")
    if args.stop_at_step is not None and not 1 <= args.stop_at_step <= args.steps:
        parser.error("stop-at-step must lie in the requested total-step horizon")
    if args.resume is None and (args.output_dir / "checkpoint.pt").exists():
        parser.error("Refusing to overwrite an existing run; choose a fresh directory or explicitly resume")
    if args.resume is not None and args.warm_start is not None:
        parser.error("Use resume or warm-start, not both")
    if args.local_channels < 1:
        parser.error("local-channels must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = choose_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")
    if sha256_file(args.initial_checkpoint) != INITIAL_SHA256:
        raise ValueError("This continuation requires the preserved original pure-VAE initializer")
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    if args.architecture == "spatial":
        from spatial_vae import SpatialVAE
        model = SpatialVAE(image_size=128, global_dim=128, local_channels=args.local_channels, posterior_mode=args.spatial_posterior_mode).to(device)
        model.base.load_state_dict(initial["model_state_dict"], strict=True)
    else:
        model = ConvVAE(image_size=128, latent_dim=128).to(device)
        model.load_state_dict(initial["model_state_dict"], strict=True)
    warm_provenance = None
    if args.warm_start is not None:
        warm = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        if warm.get("model_type") != "pure_vae" or warm.get("initial_checkpoint_sha256") != INITIAL_SHA256:
            raise ValueError("Warm-start weights must have verified pure-VAE ancestry")
        source_architecture = warm["config"].get("architecture", "conv")
        if args.architecture == "spatial" and source_architecture == "conv":
            model.base.load_state_dict(warm["model_state_dict"], strict=True)
        elif source_architecture == args.architecture:
            model.load_state_dict(warm["model_state_dict"], strict=True)
        else:
            raise ValueError("Unsupported pure-VAE architecture transition")
        warm_provenance = {"checkpoint_sha256": sha256_file(args.warm_start), "checkpoint": str(args.warm_start), "source_architecture": source_architecture, "source_step": warm["step"], "source_training_provenance": warm["training_provenance"], "optimizer": "Fresh optimizer for explicit warm-start"}
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    batch_generator = torch.Generator(device="cpu").manual_seed(args.seed + 300)
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed + 301)
    dataset = ImageFolder(args.data_root / "train", transform=make_transform(128, training=False))
    count = len(dataset) if args.max_train_images is None else min(args.max_train_images, len(dataset))
    if args.max_train_images is None and count != 5216:
        raise ValueError("Expected all5216 official training images from the pinned dataset")
    if args.batch_size > count:
        raise ValueError("Batch size exceeds the training set")
    objective_label = "L1+SSIM reconstruction+Gaussian KL" if args.ssim_weight else "L1+Gaussian KL"
    print(f"Loading {count} official-training images to {device}; learned Gaussian posterior, {objective_label}.", flush=True)
    images = torch.stack([dataset[index][0] for index in range(count)]).to(device)
    manifest_path = ROOT / "experiments/diagnostics_v1/data_manifest.json"
    common = json.loads(manifest_path.read_text())["splits"]["validation"]["indices"] if manifest_path.is_file() else []
    monitor_indices = [index for index in common if index < count][:64]
    if not monitor_indices:
        monitor_indices = list(range(min(64, count)))
    monitor_images = images[torch.tensor(monitor_indices, device=device)]
    order = torch.randperm(count, generator=batch_generator)
    position, start_step, elapsed_before = 0, 0, 0.0
    history, segments = [], []
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("model_type") != "pure_vae" or saved.get("initial_checkpoint_sha256") != INITIAL_SHA256:
            raise ValueError("Resume checkpoint must belong to this pure-VAE ancestry")
        previous = saved["training_configuration"]
        if previous.get("architecture", "conv") != args.architecture or (args.architecture == "spatial" and previous.get("local_channels") != args.local_channels):
            raise ValueError("Resume architecture must match the saved pure VAE")
        if args.architecture == "spatial" and previous.get("spatial_posterior_mode", "residual") != args.spatial_posterior_mode:
            raise ValueError("Resume must preserve spatial posterior coordinates; use an explicit fresh warm-start transition")
        if any(previous.get(name) != getattr(args, name) for name in ("seed", "batch_size", "max_train_images")):
            raise ValueError("Resume must preserve seed, batch size and training subset")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        batch_generator.set_state(saved["batch_generator_state"])
        noise_generator.set_state(saved["noise_generator_state"])
        order, position = saved["batch_order"], saved["batch_position"]
        start_step, elapsed_before = saved["step"], saved["training_seconds"]
        history = [row for row in json.loads((args.resume.parent / "history.json").read_text()) if row["step"] <= start_step]
        segments = saved["training_provenance"]["continuation_segments"]
        warm_provenance = saved["training_provenance"].get("warm_start")
    if start_step >= args.steps or (args.stop_at_step is not None and start_step >= args.stop_at_step):
        raise ValueError("The requested stop must follow the restored checkpoint")
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config.update({"image_size": 128, "latent_dim": model.latent_dim, "global_latent_dim": 128, "local_size": 16 if args.architecture == "spatial" else 0, "training_images": count, "objective": "sum(abs(x-reconstruction))/batch + beta * analytic Gaussian KL/batch", "augmentation": "none", "posterior": "Learned diagonal Gaussian mean and log-variance", "prior": "N(0,I)", "initial_checkpoint_sha256": INITIAL_SHA256, "monitor_indices": monitor_indices, "monitor_scope": "Train-seen; initializer already used every official training image"})
    if args.architecture == "spatial" and args.spatial_posterior_mode == "conditional_absolute":
        config.update({"posterior": "Conditional learned Gaussian posterior: global code and whitened local code given the sampled global code", "kl_definition": "Analytic KL(global posterior||N(0,I)) plus sampled-global expectation of analytic KL(absolute local posterior||conditional Gaussian decoder distribution)", "monitor_latent_scope": "Whitened local posterior at the global posterior mean; these are conditional diagnostics, not full aggregate-posterior moments"})
    if args.ssim_weight:
        config.update({"objective": "L1+SSIM reconstruction+Gaussian KL: sum(abs(x-reconstruction))/batch + beta * analytic Gaussian KL/batch + effective_ssim_weight * image_pixels * mean(1-SSIM)", "ssim_definition": "Single-scale11x11 Gaussian window, sigma1.5, valid convolution, data_range1, K1=.01,K2=.03; mean per-image SSIM", "ssim_weight_schedule": "target * min(1,(step-starting_after_step)/max(1,ssim_ramp)); no ramp when ssim_ramp=0"})
    segments.append({"starting_after_step": start_step, "configuration": config, "resume_checkpoint_sha256": sha256_file(args.resume) if args.resume is not None else None})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "process.json", {"pid": os.getpid(), "status": "running"})
    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "segments.json", segments)
    write_json(args.output_dir / "training_manifest.json", {"count": count, "files": [Path(dataset.samples[index][0]).relative_to(dataset.root).as_posix() for index in range(count)], "split": "official train", "initial_checkpoint_sha256": INITIAL_SHA256, "monitor_indices": monitor_indices})
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    start_time = time.perf_counter()
    interval = []
    initial_monitor = save_monitor(model, monitor_images, args.output_dir, start_step, args.batch_size)
    if not history or history[-1]["step"] != start_step:
        history.append({"step": start_step, "monitor": initial_monitor, "training_seconds": elapsed_before})
    write_json(args.output_dir / "history.json", history)

    def save_checkpoint(step: int, monitor: dict) -> None:
        elapsed = elapsed_before + time.perf_counter() - start_time
        provenance = {"model_type": "pure_vae", "initializer_sha256": INITIAL_SHA256, "initializer_training_images": 5216, "continuation_training_images": count, "continuation_steps": step, "continuation_segments": segments, "warm_start": warm_provenance, "monitor_status": config["monitor_scope"]}
        version = ("pure-vae-l1-ssim-kl-v2" if args.ssim_weight else "pure-vae-l1-kl-v1") + ("-spatial" if args.architecture == "spatial" else "") + ("-conditional" if args.architecture == "spatial" and args.spatial_posterior_mode == "conditional_absolute" else "")
        weights = {"model_state_dict": cpu_copy(model.state_dict()), "config": config, "training_configuration": config, "training_provenance": provenance, "model_type": "pure_vae", "model_version": version, "initial_checkpoint_sha256": INITIAL_SHA256, "step": step, "training_seconds": elapsed, "monitor": monitor, "selection_note": "Training/visual monitoring only; no official test scores used during training"}
        atomic_save(args.output_dir / f"model_step_{step:06d}.pt", weights)
        atomic_save(args.output_dir / "checkpoint.pt", {**weights, "optimizer_state_dict": cpu_copy(optimizer.state_dict()), "batch_generator_state": batch_generator.get_state(), "noise_generator_state": noise_generator.get_state(), "batch_order": order, "batch_position": position})

    step = start_step
    try:
        for step in range(start_step + 1, args.steps + 1):
            if position == count:
                order, position = torch.randperm(count, generator=batch_generator), 0
            indices = order[position:min(position + args.batch_size, count)]
            position += len(indices)
            batch = images[indices.to(device)]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = args.min_learning_rate + 0.5 * (args.learning_rate - args.min_learning_rate) * (1 + math.cos(math.pi * (step - 1) / args.steps))
            for group in optimizer.param_groups:
                group["lr"] = lr
            beta_fraction = min(1.0, (step - start_step) / max(1, args.beta_ramp)) if args.beta_ramp else 1.0
            beta = args.beta_start + beta_fraction * (args.beta - args.beta_start)
            ssim_fraction = min(1.0, (step - start_step) / max(1, args.ssim_ramp)) if args.ssim_ramp else 1.0
            ssim_weight = args.ssim_weight * ssim_fraction
            if args.architecture == "spatial" and args.spatial_posterior_mode == "conditional_absolute":
                noise = torch.randn((len(batch), model.latent_dim), generator=noise_generator, dtype=batch.dtype).to(device)
                reconstruction, mu, logvar = model.forward_with_noise(batch, noise)
            else:
                mu, logvar = model.encode(batch)
                noise = torch.randn(mu.shape, generator=noise_generator, dtype=mu.dtype).to(device)
                reconstruction = model.decode(mu + (0.5 * logvar).exp() * noise)
            loss, l1, kl, structural = pure_vae_objective(reconstruction, batch, mu, logvar, beta, ssim_weight)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite pure-VAE objective at step{step}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not torch.isfinite(gradient_norm).item():
                raise FloatingPointError(f"Nonfinite pure-VAE gradients at step{step}")
            optimizer.step()
            values = {"loss": float(loss.detach().cpu()), "l1_sum": float(l1.detach().cpu()), "raw_kl": float(kl.detach().cpu()), "beta": beta, "learning_rate": lr, "gradient_norm": float(gradient_norm.detach().cpu())}
            if structural is not None:
                values.update({"ssim_loss": float(structural.detach().cpu()), "ssim_weight": ssim_weight})
            interval.append(values)
            due = step % args.save_every == 0 or step == args.steps or step == args.stop_at_step or stop_requested or (args.output_dir / "STOP_AFTER_CHECKPOINT").exists()
            if step % args.log_every == 0 or due:
                row = {"step": step, "interval_steps": len(interval), "training_seconds": elapsed_before + time.perf_counter() - start_time, **{name: sum(item[name] for item in interval) / len(interval) for name in interval[0]}}
                interval.clear()
                if due:
                    row["monitor"] = save_monitor(model, monitor_images, args.output_dir, step, args.batch_size)
                history.append(row)
                write_json(args.output_dir / "history.json", history)
                print(json.dumps({key: value for key, value in row.items() if key != "monitor"}), flush=True)
                if due:
                    save_checkpoint(step, row["monitor"])
            del batch, mu, logvar, noise, reconstruction, loss, l1, kl, structural, gradient_norm
            if device.type == "mps" and (step % 25 == 0 or due):
                torch.mps.empty_cache()
            if stop_requested or step == args.stop_at_step or (args.output_dir / "STOP_AFTER_CHECKPOINT").exists():
                break
    except BaseException:
        write_json(args.output_dir / "process.json", {"pid": os.getpid(), "status": "error", "last_attempted_step": step})
        raise
    write_json(args.output_dir / "process.json", {"pid": os.getpid(), "status": "complete" if step == args.steps else "stopped", "last_step": step})
    print(f"Finished pure VAE at step{step}; artifacts in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

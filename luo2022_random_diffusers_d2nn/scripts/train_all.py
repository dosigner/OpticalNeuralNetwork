"""Train all models needed for figure reproduction.

Optimized for A100 GPU: AMP, torch.compile, adaptive batch size.
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import torch
import torch.utils.data

from luo2022_d2nn.config.schema import load_and_validate_config
from luo2022_d2nn.data.mnist import MNISTAmplitude
from luo2022_d2nn.models.d2nn import D2NN
from luo2022_d2nn.optics.bl_asm import bl_asm_propagate, bl_asm_transfer_function
from luo2022_d2nn.diffuser.random_phase import generate_diffuser
from luo2022_d2nn.training.losses import pcc_energy_loss, pearson_correlation
from luo2022_d2nn.training.schedules import build_scheduler
from luo2022_d2nn.utils.io import dump_yaml, dump_json
from luo2022_d2nn.utils.seed import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _unpack_batch(batch, device):
    if isinstance(batch, dict):
        amp = batch["amplitude"]
        mask = batch["mask"]
        target = amp
    else:
        amp, target, mask = batch[0], batch[1], batch[2]
    amp = amp.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    if amp.ndim == 4 and amp.shape[1] == 1:
        amp = amp.squeeze(1)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    return amp, target, mask


def build_model(cfg: dict, device: torch.device) -> D2NN:
    geom = cfg["geometry"]
    grid = cfg["grid"]
    optics = cfg["optics"]
    model_cfg = cfg["model"]
    model = D2NN(
        num_layers=int(geom["num_layers"]),
        grid_size=int(grid["nx"]),
        dx_mm=float(grid["pitch_mm"]),
        wavelength_mm=float(optics["wavelength_mm"]),
        diffuser_to_layer1_mm=float(geom["diffuser_to_layer1_mm"]),
        layer_to_layer_mm=float(geom["layer_to_layer_mm"]),
        last_layer_to_output_mm=float(geom["last_layer_to_output_mm"]),
        pad_factor=int(grid.get("pad_factor", 2)),
        init_phase_dist=str(model_cfg.get("init_phase_distribution", "uniform_0_2pi")),
    )
    return model.to(device)


def generate_diffusers(n, grid_size, dx_mm, wavelength_mm, epoch_seed, device, diff_params):
    """Generate n diffuser transmittances, return stacked tensor (n, N, N)."""
    transmittances = []
    for i in range(n):
        seed = epoch_seed * 1000 + i
        result = generate_diffuser(
            grid_size, dx_mm, wavelength_mm,
            seed=seed, device=device, **diff_params,
        )
        transmittances.append(result["transmittance"])
    return torch.stack(transmittances, dim=0)  # (n, N, N)


def apply_diffusers_vectorized(amplitude, diffuser_stack, H_obj_to_diff, pad_factor):
    """Vectorized diffuser application: (B,N,N) × (n,N,N) → (B*n,N,N)."""
    B = amplitude.shape[0]
    n = diffuser_stack.shape[0]
    N = amplitude.shape[-1]

    field = amplitude.to(torch.complex64)
    field_at_diff = bl_asm_propagate(field, H_obj_to_diff, pad_factor=pad_factor)
    # (B,1,N,N) * (1,n,N,N) → (B,n,N,N)
    result = field_at_diff.unsqueeze(1) * diffuser_stack.unsqueeze(0)
    return result.reshape(B * n, N, N)


def train_single(cfg: dict, run_name: str, device: torch.device, runs_root: Path, batch_size: int = 64):
    """Train a single model with GPU optimizations."""
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = run_dir / "model.pt"
    if ckpt_path.exists():
        logger.info("Checkpoint exists: %s — skipping", ckpt_path)
        return ckpt_path

    seed = int(cfg["experiment"]["seed"])
    set_global_seed(seed, deterministic=False)
    dump_yaml(run_dir / "config.yaml", cfg)

    # Dataset
    ds_cfg = cfg["dataset"]
    data_root = str(runs_root.parent / "data")
    train_ds = MNISTAmplitude(
        root=data_root, split="train",
        resize_to=int(ds_cfg.get("resize_to_px", 160)),
        final_size=int(ds_cfg.get("final_resolution_px", 240)),
    )

    n = int(cfg["training"]["diffusers_per_epoch"])
    B = batch_size
    logger.info("[%s] Using batch_size=%d (B*n=%d)", run_name, B, B * n)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=B, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True,
        persistent_workers=True,
    )

    # Model
    model = build_model(cfg, device)
    logger.info("[%s] %d layers, %d params", run_name, model.num_layers,
                sum(p.numel() for p in model.parameters()))

    compiled_model = model  # torch.compile doesn't support complex ops

    # Optimizer + scheduler
    tr = cfg["training"]
    lr = float(tr["learning_rate_initial"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = build_scheduler(optimizer, tr["learning_rate_schedule"])

    alpha = float(tr["loss"].get("alpha", 1.0))
    beta = float(tr["loss"].get("beta", 0.5))

    # Optics params
    grid = cfg["grid"]
    grid_size = int(grid["nx"])
    dx_mm = float(grid["pitch_mm"])
    wavelength_mm = float(cfg["optics"]["wavelength_mm"])
    pad_factor = int(grid.get("pad_factor", 2))
    obj_to_diff_mm = float(cfg["geometry"]["object_to_diffuser_mm"])
    base_seed = int(cfg["experiment"]["seed"])

    diff_params = {
        "delta_n": float(cfg["diffuser"].get("delta_n", 0.74)),
        "height_mean_lambda": float(cfg["diffuser"].get("height_mean_lambda", 25.0)),
        "height_std_lambda": float(cfg["diffuser"].get("height_std_lambda", 8.0)),
        "smoothing_sigma_lambda": float(cfg["diffuser"].get("smoothing_sigma_lambda", 4.0)),
    }

    # No AMP — complex-valued ops don't support autocast well
    use_amp = False

    epochs = int(tr["epochs"])
    t0 = time.time()

    H_obj_to_diff = bl_asm_transfer_function(
        grid_size, dx_mm, wavelength_mm, obj_to_diff_mm,
        pad_factor=pad_factor, device=device,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_seed = base_seed + epoch

        # Generate diffusers for this epoch
        diffuser_stack = generate_diffusers(
            n, grid_size, dx_mm, wavelength_mm, epoch_seed, device, diff_params,
        )

        total_loss = 0.0
        total_pcc = 0.0
        num_batches = 0

        for batch in train_loader:
            amplitude, target, mask = _unpack_batch(batch, device)
            batch_B = amplitude.shape[0]

            # Apply diffusers vectorized
            field_after_diff = apply_diffusers_vectorized(
                amplitude, diffuser_stack, H_obj_to_diff, pad_factor,
            )

            # Forward
            output_field = compiled_model(field_after_diff)
            output_intensity = output_field.abs() ** 2

            # Duplicate target/mask
            target_dup = target.unsqueeze(1).expand(batch_B, n, -1, -1).reshape(batch_B * n, *target.shape[1:])
            mask_dup = mask.unsqueeze(1).expand(batch_B, n, -1, -1).reshape(batch_B * n, *mask.shape[1:])

            loss = pcc_energy_loss(output_intensity, target_dup, mask_dup, alpha, beta)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pcc_val = pearson_correlation(output_intensity.float(), target_dup).item()
            total_loss += loss.item()
            total_pcc += pcc_val
            num_batches += 1

        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_pcc = total_pcc / max(num_batches, 1)
        elapsed = time.time() - t0

        logger.info(
            "[%s] Epoch %d/%d — loss: %.5f, pcc: %.4f, lr: %.2e (%.0fs)",
            run_name, epoch, epochs, avg_loss, avg_pcc,
            optimizer.param_groups[0]["lr"], elapsed,
        )

        if epoch % 10 == 0 or epoch == epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
                "metrics": {"avg_loss": avg_loss, "avg_pcc": avg_pcc},
            }, ckpt_path)

    total_time = time.time() - t0
    logger.info("[%s] Done in %.1f min", run_name, total_time / 60)
    dump_json(run_dir / "training_summary.json", {
        "run_name": run_name,
        "epochs": epochs,
        "final_loss": avg_loss,
        "final_pcc": avg_pcc,
        "total_time_s": total_time,
    })
    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs-root", default="runs", type=str)
    parser.add_argument("--smoke-test", action="store_true", help="2 epochs only")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Object batch size (default: config batch_size_objects, then 64)")
    parser.add_argument("--core-only", action="store_true",
                        help="Only train 4-layer models (skip depth sweep)")
    args = parser.parse_args()

    device = torch.device(args.device)
    runs_root = Path(args.runs_root)
    base_cfg = load_and_validate_config(args.config)
    epochs = 2 if args.smoke_test else args.epochs

    # Resolve batch size: CLI > config > default(64)
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = int(base_cfg.get("training", {}).get("batch_size_objects", 64))
    logger.info("Batch size: %d", batch_size)

    # A100 optimizations
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info("TF32 enabled, matmul precision=high")

    runs = [
        ("n1_L4", 4, 1),
        ("n10_L4", 4, 10),
        ("n15_L4", 4, 15),
        ("n20_L4", 4, 20),
    ]
    if not args.core_only:
        runs += [
            ("n1_L2", 2, 1), ("n10_L2", 2, 10),
            ("n15_L2", 2, 15), ("n20_L2", 2, 20),
            ("n1_L5", 5, 1), ("n10_L5", 5, 10),
            ("n15_L5", 5, 15), ("n20_L5", 5, 20),
        ]

    if args.only:
        runs = [(name, layers, n) for name, layers, n in runs if name == args.only]

    checkpoints = {}
    for run_name, num_layers, n_diffusers in runs:
        logger.info("=" * 60)
        logger.info("Training: %s (L=%d, n=%d, epochs=%d)", run_name, num_layers, n_diffusers, epochs)

        cfg = copy.deepcopy(base_cfg)
        cfg["experiment"]["id"] = run_name
        cfg["geometry"]["num_layers"] = num_layers
        cfg["model"]["num_layers"] = num_layers
        cfg["training"]["diffusers_per_epoch"] = n_diffusers
        cfg["training"]["epochs"] = epochs

        ckpt = train_single(cfg, run_name, device, runs_root, batch_size=batch_size)
        checkpoints[run_name] = str(ckpt)
        torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("All done!")
    for name, path in checkpoints.items():
        logger.info("  %s → %s", name, path)
    dump_json(runs_root / "checkpoints.json", checkpoints)


if __name__ == "__main__":
    main()

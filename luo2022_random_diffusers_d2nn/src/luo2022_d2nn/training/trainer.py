"""Training loop with B x n diffuser logic (Luo et al. 2022)."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from luo2022_d2nn.diffuser.random_phase import generate_diffuser
from luo2022_d2nn.models.d2nn import D2NN
from luo2022_d2nn.optics.bl_asm import bl_asm_propagate, bl_asm_transfer_function
from luo2022_d2nn.training.losses import pcc_energy_loss, pearson_correlation
from luo2022_d2nn.training.schedules import build_scheduler

logger = logging.getLogger(__name__)


def _unpack_batch(batch, device: torch.device):
    """Extract (amplitude, target, mask) tensors from a batch.

    Supports dict-style batches (from MNISTAmplitude) and tuple/list batches.
    For dict batches, amplitude serves as both input and target.
    Squeezes channel dim if present: (B, 1, N, N) → (B, N, N).
    """
    if isinstance(batch, dict):
        amp = batch["amplitude"]
        mask = batch["mask"]
        target = amp  # target is the amplitude itself
    elif isinstance(batch, (list, tuple)):
        amp, target, mask = batch[0], batch[1], batch[2]
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    amp = amp.to(device)
    target = target.to(device)
    mask = mask.to(device)

    # Squeeze channel dim: (B, 1, N, N) -> (B, N, N)
    if amp.ndim == 4 and amp.shape[1] == 1:
        amp = amp.squeeze(1)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)

    return amp, target, mask


class Trainer:
    """D2NN trainer with epoch-level diffuser generation."""

    def __init__(self, model: D2NN, cfg: dict[str, Any], device: torch.device) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

        tr = cfg["training"]
        self.n_diffusers = int(tr["diffusers_per_epoch"])

        # Loss parameters
        loss_cfg = tr["loss"]
        self.alpha = float(loss_cfg.get("alpha", 1.0))
        self.beta = float(loss_cfg.get("beta", 0.5))

        # Optimizer
        lr = float(tr["learning_rate_initial"])
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Scheduler
        self.scheduler = build_scheduler(self.optimizer, tr["learning_rate_schedule"])

        # Grid / optics parameters for object-to-diffuser propagation
        grid = cfg["grid"]
        self.grid_size = int(grid["nx"])
        self.dx_mm = float(grid["pitch_mm"])
        self.wavelength_mm = float(cfg["optics"]["wavelength_mm"])
        self.pad_factor = int(grid.get("pad_factor", 2))

        # Geometry
        geom = cfg["geometry"]
        self.object_to_diffuser_mm = float(geom["object_to_diffuser_mm"])

        # Diffuser generation parameters
        diff_cfg = cfg["diffuser"]
        self.diffuser_params = {
            "delta_n": float(diff_cfg.get("delta_n", 0.74)),
            "height_mean_lambda": float(diff_cfg.get("height_mean_lambda", 25.0)),
            "height_std_lambda": float(diff_cfg.get("height_std_lambda", 8.0)),
            "smoothing_sigma_lambda": float(diff_cfg.get("smoothing_sigma_lambda", 4.0)),
        }

        # Base seed
        self.base_seed = int(cfg["experiment"]["seed"])

    # ------------------------------------------------------------------
    def _generate_epoch_diffusers(self, n: int, epoch_seed: int) -> list[torch.Tensor]:
        """Generate n diffuser transmittances for this epoch.

        Uses generate_diffuser() with seed = epoch_seed * 1000 + i for diffuser i.
        Returns list of complex transmittance tensors, shape (N, N).
        """
        diffusers = []
        for i in range(n):
            seed = epoch_seed * 1000 + i
            result = generate_diffuser(
                self.grid_size,
                self.dx_mm,
                self.wavelength_mm,
                seed=seed,
                device=self.device,
                **self.diffuser_params,
            )
            diffusers.append(result["transmittance"])
        return diffusers

    # ------------------------------------------------------------------
    def _apply_diffusers(
        self,
        amplitude: torch.Tensor,
        diffusers: list[torch.Tensor],
        H_obj_to_diff: torch.Tensor,
    ) -> torch.Tensor:
        """Apply diffusers to object amplitudes.

        For each object in batch (B, N, N) and each diffuser (n total):
        1. Propagate amplitude (as plane wave) from object to diffuser plane
        2. Multiply by diffuser transmittance

        Returns
        -------
        Tensor, shape (B*n, N, N) — complex field at diffuser output.
        """
        B = amplitude.shape[0]
        n = len(diffusers)

        # Input field: amplitude as complex plane wave
        field = amplitude.to(torch.complex64)  # (B, N, N)

        # Propagate object to diffuser plane
        field_at_diff = bl_asm_propagate(field, H_obj_to_diff, pad_factor=self.pad_factor)
        # field_at_diff: (B, N, N)

        # Duplicate across n diffusers and apply each
        outputs = []
        for d_idx in range(n):
            t_d = diffusers[d_idx]  # (N, N)
            # Multiply each sample in batch by this diffuser
            outputs.append(field_at_diff * t_d.unsqueeze(0))  # (B, N, N)

        # Stack: (n, B, N, N) -> reshape to (B*n, N, N)
        stacked = torch.stack(outputs, dim=0)  # (n, B, N, N)
        result = stacked.permute(1, 0, 2, 3).reshape(B * n, self.grid_size, self.grid_size)
        return result

    # ------------------------------------------------------------------
    def train_epoch(self, dataloader: Any, epoch: int) -> dict[str, float]:
        """Train one epoch with B x n diffuser logic.

        Steps:
        1. Generate n diffusers (fixed for entire epoch)
        2. Pre-compute H_obj_to_diff transfer function
        3. For each batch: duplicate x n, apply diffusers, forward, loss, backward
        4. Step scheduler once at epoch end

        Returns dict with avg_loss, avg_pcc.
        """
        self.model.train()
        epoch_seed = self.base_seed + epoch

        # 1. Generate diffusers for this epoch
        diffusers = self._generate_epoch_diffusers(self.n_diffusers, epoch_seed)

        # 2. Pre-compute object-to-diffuser transfer function
        H_obj_to_diff = bl_asm_transfer_function(
            self.grid_size,
            self.dx_mm,
            self.wavelength_mm,
            self.object_to_diffuser_mm,
            pad_factor=self.pad_factor,
            device=self.device,
        )

        n = self.n_diffusers
        total_loss = 0.0
        total_pcc = 0.0
        num_batches = 0

        for batch in dataloader:
            # Expect batch to be (amplitude, target, mask) or similar
            amplitude, target, mask = _unpack_batch(batch, self.device)
            B = amplitude.shape[0]

            # a. Apply diffusers: B objects x n diffusers -> B*n fields
            field_after_diff = self._apply_diffusers(amplitude, diffusers, H_obj_to_diff)

            # b. Forward through D2NN
            output_field = self.model(field_after_diff)  # (B*n, N, N)

            # c. Compute intensity
            output_intensity = (output_field.abs()) ** 2  # (B*n, N, N)

            # d. Duplicate target and mask n times: (B, N, N) -> (B*n, N, N)
            target_dup = target.unsqueeze(1).expand(B, n, *target.shape[1:]).reshape(B * n, *target.shape[1:])
            mask_dup = mask.unsqueeze(1).expand(B, n, *mask.shape[1:]).reshape(B * n, *mask.shape[1:])

            # e. Compute loss
            loss = pcc_energy_loss(output_intensity, target_dup, mask_dup, self.alpha, self.beta)

            # f. Backward + optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            with torch.no_grad():
                pcc_val = pearson_correlation(output_intensity, target_dup).item()
            total_loss += loss.item()
            total_pcc += pcc_val
            num_batches += 1

        # Step scheduler once per epoch
        self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_pcc = total_pcc / max(num_batches, 1)

        logger.info(
            "Epoch %d: loss=%.6f, pcc=%.4f, lr=%.2e",
            epoch,
            avg_loss,
            avg_pcc,
            self.optimizer.param_groups[0]["lr"],
        )

        return {"avg_loss": avg_loss, "avg_pcc": avg_pcc}

    # ------------------------------------------------------------------
    def evaluate(
        self,
        dataloader: Any,
        diffusers: list[torch.Tensor] | None = None,
        epoch: int = 0,
    ) -> dict[str, Any]:
        """Evaluate with given diffusers (or generate new ones).

        Returns dict with mean_pcc, per_sample_pcc, etc.
        """
        self.model.eval()

        if diffusers is None:
            eval_seed = self.base_seed + 999999 + epoch
            diffusers = self._generate_epoch_diffusers(self.n_diffusers, eval_seed)

        H_obj_to_diff = bl_asm_transfer_function(
            self.grid_size,
            self.dx_mm,
            self.wavelength_mm,
            self.object_to_diffuser_mm,
            pad_factor=self.pad_factor,
            device=self.device,
        )

        n = len(diffusers)
        all_pccs: list[float] = []

        with torch.no_grad():
            for batch in dataloader:
                amplitude, target, mask = _unpack_batch(batch, self.device)
                B = amplitude.shape[0]

                field_after_diff = self._apply_diffusers(amplitude, diffusers, H_obj_to_diff)
                output_field = self.model(field_after_diff)
                output_intensity = output_field.abs() ** 2

                target_dup = target.unsqueeze(1).expand(B, n, *target.shape[1:]).reshape(B * n, *target.shape[1:])
                mask_dup = mask.unsqueeze(1).expand(B, n, *mask.shape[1:]).reshape(B * n, *mask.shape[1:])

                from luo2022_d2nn.eval.pcc import compute_pcc
                pcc_vals = compute_pcc(output_intensity, target_dup)
                all_pccs.extend(pcc_vals.cpu().tolist())

        mean_pcc = sum(all_pccs) / max(len(all_pccs), 1)
        return {
            "mean_pcc": mean_pcc,
            "per_sample_pcc": all_pccs,
            "num_samples": len(all_pccs),
        }

"""Evaluate a trained beam-cleanup model and save summary artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from kim2026.cli.common import apply_runtime_environment, choose_device, dump_json, load_config
from kim2026.data.dataset import CachedFieldDataset
from kim2026.models.d2nn import BeamCleanupD2NN
from kim2026.models.fd2nn import BeamCleanupFD2NN
from kim2026.training.metrics import beam_cleanup_selection_summary, summarize_metrics
from kim2026.training.targets import apply_receiver_aperture, make_detector_plane_target
from kim2026.utils.seed import set_global_seed
from kim2026.viz.beam_plots import save_triptych


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "u_vacuum": torch.stack([item["u_vacuum"] for item in batch], dim=0),
        "u_turb": torch.stack([item["u_turb"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }


def _build_eval_model(cfg: dict[str, Any], sample_n: int, device: torch.device) -> torch.nn.Module:
    model_type = str(cfg["model"].get("type", "d2nn"))
    if model_type == "fd2nn":
        dual_2f = cfg["optics"]["dual_2f"]
        model = BeamCleanupFD2NN(
            n=sample_n,
            wavelength_m=float(cfg["optics"]["lambda_m"]),
            window_m=float(cfg["grid"]["receiver_window_m"]),
            num_layers=int(cfg["model"]["num_layers"]),
            layer_spacing_m=float(cfg["model"].get("layer_spacing_m", 0.0)),
            phase_max=float(cfg["model"].get("phase_max", 3.14159265)),
            phase_constraint=str(cfg["model"].get("phase_constraint", "unconstrained")),
            phase_init=str(cfg["model"].get("phase_init", "uniform")),
            phase_init_scale=float(cfg["model"].get("phase_init_scale", 0.1)),
            dual_2f_f1_m=float(dual_2f["f1_m"]),
            dual_2f_f2_m=float(dual_2f["f2_m"]),
            dual_2f_na1=None if dual_2f.get("na1") is None else float(dual_2f["na1"]),
            dual_2f_na2=None if dual_2f.get("na2") is None else float(dual_2f["na2"]),
            dual_2f_apply_scaling=bool(dual_2f.get("apply_scaling", False)),
        )
    else:
        model = BeamCleanupD2NN(
            n=sample_n,
            wavelength_m=float(cfg["optics"]["lambda_m"]),
            window_m=float(cfg["grid"]["receiver_window_m"]),
            num_layers=int(cfg["model"]["num_layers"]),
            layer_spacing_m=float(cfg["model"]["layer_spacing_m"]),
            detector_distance_m=float(cfg["model"]["detector_distance_m"]),
        )
    return model.to(device)


def _split_complex(field: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    array = field.detach().cpu().numpy()
    return array.real.astype(np.float32), array.imag.astype(np.float32)


def _save_sample_fields(
    path: str | Path,
    *,
    input_field: torch.Tensor,
    vacuum_field: torch.Tensor,
    baseline_field: torch.Tensor,
    pred_field: torch.Tensor,
    target_field: torch.Tensor,
) -> None:
    input_real, input_imag = _split_complex(input_field)
    vacuum_real, vacuum_imag = _split_complex(vacuum_field)
    baseline_real, baseline_imag = _split_complex(baseline_field)
    pred_real, pred_imag = _split_complex(pred_field)
    target_real, target_imag = _split_complex(target_field)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input_real=input_real,
        input_imag=input_imag,
        vacuum_real=vacuum_real,
        vacuum_imag=vacuum_imag,
        baseline_real=baseline_real,
        baseline_imag=baseline_imag,
        pred_real=pred_real,
        pred_imag=pred_imag,
        target_real=target_real,
        target_imag=target_imag,
    )


def _resolve_checkpoint_path(cfg: dict[str, Any], checkpoint_path: str | Path | None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    return Path(cfg["experiment"]["save_dir"]) / "checkpoint.pt"


def _load_model_state(model: torch.nn.Module, checkpoint_path: Path, *, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override")
    parser.add_argument("--split", default=None, help="Override evaluation split")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    apply_runtime_environment(cfg["runtime"])
    set_global_seed(int(cfg["runtime"]["seed"]), strict_reproducibility=bool(cfg["runtime"]["strict_reproducibility"]))
    device = choose_device(cfg["runtime"])

    split = args.split or str(cfg["evaluation"].get("split", "test"))
    dataset = CachedFieldDataset(
        cache_dir=cfg["data"]["cache_dir"],
        manifest_path=cfg["data"]["split_manifest_path"],
        split=split,
    )
    if len(dataset) == 0:
        raise ValueError(f"evaluation dataset is empty for split '{split}'")

    sample_n = int(dataset[0]["u_turb"].shape[-1])
    model = _build_eval_model(cfg, sample_n=sample_n, device=device)
    checkpoint_path = _resolve_checkpoint_path(cfg, args.checkpoint)
    _load_model_state(model, checkpoint_path, device=device)
    model.eval()

    runtime_cfg = cfg["runtime"]
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])),
        "shuffle": False,
        "num_workers": int(runtime_cfg.get("num_workers", 0)),
        "pin_memory": bool(runtime_cfg.get("pin_memory", False)),
        "persistent_workers": bool(runtime_cfg.get("persistent_workers", False)),
        "collate_fn": _collate,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = int(runtime_cfg.get("prefetch_factor", 2))
    else:
        loader_kwargs.pop("persistent_workers")
    loader = DataLoader(dataset, **loader_kwargs)

    receiver_window_m = float(cfg["grid"]["receiver_window_m"])
    aperture_diameter_m = float(cfg["receiver"]["aperture_diameter_m"])
    wavelength_m = float(cfg["optics"]["lambda_m"])
    model_type = str(cfg["model"].get("type", "d2nn"))
    loss_mode = str(cfg["training"]["loss"].get("mode", "intensity"))
    complex_mode = loss_mode in ("complex", "roi_complex")
    total_distance_m = 0.0 if model_type == "fd2nn" else (
        (int(cfg["model"]["num_layers"]) - 1) * float(cfg["model"]["layer_spacing_m"])
        + float(cfg["model"]["detector_distance_m"])
    )

    pred_fields: list[torch.Tensor] = []
    baseline_fields: list[torch.Tensor] = []
    target_fields: list[torch.Tensor] = []

    first_input: torch.Tensor | None = None
    first_vacuum: torch.Tensor | None = None
    first_baseline: torch.Tensor | None = None
    first_pred: torch.Tensor | None = None
    first_target: torch.Tensor | None = None

    with torch.no_grad():
        for batch in loader:
            u_turb = batch["u_turb"].to(device)
            u_vacuum = batch["u_vacuum"].to(device)

            input_field = apply_receiver_aperture(
                u_turb,
                receiver_window_m=receiver_window_m,
                aperture_diameter_m=aperture_diameter_m,
            )
            vacuum_field = apply_receiver_aperture(
                u_vacuum,
                receiver_window_m=receiver_window_m,
                aperture_diameter_m=aperture_diameter_m,
            )
            baseline_field = make_detector_plane_target(
                u_turb,
                wavelength_m=wavelength_m,
                receiver_window_m=receiver_window_m,
                aperture_diameter_m=aperture_diameter_m,
                total_distance_m=total_distance_m,
                complex_mode=True,
            )
            target_field = make_detector_plane_target(
                u_vacuum,
                wavelength_m=wavelength_m,
                receiver_window_m=receiver_window_m,
                aperture_diameter_m=aperture_diameter_m,
                total_distance_m=total_distance_m,
                complex_mode=True,
            )
            pred_field = model(input_field)

            pred_fields.append(pred_field.detach().cpu())
            baseline_fields.append(baseline_field.detach().cpu())
            target_fields.append(target_field.detach().cpu())

            if first_input is None:
                first_input = input_field[0].detach().cpu()
                first_vacuum = vacuum_field[0].detach().cpu()
                first_baseline = baseline_field[0].detach().cpu()
                first_pred = pred_field[0].detach().cpu()
                first_target = target_field[0].detach().cpu()

    pred_tensor = torch.cat(pred_fields, dim=0)
    baseline_tensor = torch.cat(baseline_fields, dim=0)
    target_tensor = torch.cat(target_fields, dim=0)

    if complex_mode:
        model_metrics = summarize_metrics(pred_tensor, target_tensor, window_m=receiver_window_m, complex_mode=True)
        baseline_metrics = summarize_metrics(baseline_tensor, target_tensor, window_m=receiver_window_m, complex_mode=True)
    else:
        model_metrics = summarize_metrics(
            pred_tensor.abs().square(),
            target_tensor.abs().square(),
            window_m=receiver_window_m,
            complex_mode=False,
        )
        baseline_metrics = summarize_metrics(
            baseline_tensor.abs().square(),
            target_tensor.abs().square(),
            window_m=receiver_window_m,
            complex_mode=False,
        )

    run_dir = Path(cfg["experiment"]["save_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation = {"baseline": baseline_metrics, "model": model_metrics}
    if complex_mode:
        leakage_threshold = float(cfg["training"]["loss"].get("leakage_threshold", 0.15))
        evaluation["selection"] = {
            "strategy": "roi50_intensity_overlap_then_phase_with_leakage_gate",
            "leakage_threshold": leakage_threshold,
            "baseline": beam_cleanup_selection_summary(baseline_metrics, leakage_threshold=leakage_threshold),
            "model": beam_cleanup_selection_summary(model_metrics, leakage_threshold=leakage_threshold),
        }
    if bool(cfg["evaluation"].get("save_json", True)):
        dump_json(run_dir / "evaluation.json", evaluation)

    if bool(cfg["visualization"].get("save_raw", True)):
        assert first_input is not None
        assert first_vacuum is not None
        assert first_baseline is not None
        assert first_pred is not None
        assert first_target is not None
        _save_sample_fields(
            run_dir / "sample_fields.npz",
            input_field=first_input,
            vacuum_field=first_vacuum,
            baseline_field=first_baseline,
            pred_field=first_pred,
            target_field=first_target,
        )

    if bool(cfg["visualization"].get("save_plots", True)):
        assert first_input is not None
        assert first_vacuum is not None
        assert first_baseline is not None
        assert first_pred is not None
        assert first_target is not None
        figure_dir = Path(cfg["visualization"]["output_dir"])
        save_triptych(
            figure_dir / f"evaluation_{split}.png",
            input_field=first_input,
            vacuum_field=first_vacuum,
            baseline_field=first_baseline,
            pred_field=first_pred,
            target_field=first_target,
            title=f"Beam Cleanup Evaluation ({split})",
        )


if __name__ == "__main__":
    main()

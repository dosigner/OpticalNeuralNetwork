"""Train saliency model with spec-style config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from tao2019_fd2nn.cli.common import build_dataloaders, build_model, choose_device, create_run_dir, load_config
from tao2019_fd2nn.optics.fft2c import gamma_flip2d
from tao2019_fd2nn.training.callbacks import save_checkpoint, save_metrics, save_resolved_config
from tao2019_fd2nn.training.metrics_saliency import max_f_measure, pr_curve
from tao2019_fd2nn.training.trainer import best_epoch_index, summarize_history, train_saliency
from tao2019_fd2nn.utils.live_log import LiveLogger
from tao2019_fd2nn.utils.math import intensity
from tao2019_fd2nn.utils.seed import set_global_seed
from tao2019_fd2nn.viz.figure_factory import FigureFactory


def _collect_preds(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    eval_crop_box: tuple[int, int, int, int] | None = None,
    gamma_flip: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for fields, targets in loader:
            fields = fields.to(device)
            _I = intensity(model(fields))
            p = _I
            g = targets
            if eval_crop_box is not None:
                y0, y1, x0, x1 = eval_crop_box
                p = p[..., y0:y1, x0:x1]
                g = g[..., y0:y1, x0:x1]
            _min = p.amin(dim=(-2, -1), keepdim=True)
            _max = p.amax(dim=(-2, -1), keepdim=True)
            p = ((p - _min) / (_max - _min).clamp_min(1e-8)).cpu()
            g = g.cpu()
            if gamma_flip:
                p = gamma_flip2d(p)
            preds.append(p)
            gts.append(g)
    if not preds:
        return torch.empty(0, 1, 1), torch.empty(0, 1, 1)
    return torch.cat(preds, dim=0), torch.cat(gts, dim=0)


def _sample_triplet(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    eval_crop_box: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        for fields, targets in loader:
            fields = fields.to(device)
            _I = intensity(model(fields))
            field0 = fields[0].detach()
            gt0 = targets[0].detach()
            pred0 = _I[0].detach()
            if eval_crop_box is not None:
                y0, y1, x0, x1 = eval_crop_box
                field0 = field0[..., y0:y1, x0:x1]
                gt0 = gt0[..., y0:y1, x0:x1]
                pred0 = pred0[..., y0:y1, x0:x1]
            _min = pred0.amin(dim=(-2, -1), keepdim=True)
            _max = pred0.amax(dim=(-2, -1), keepdim=True)
            pred0 = (pred0 - _min) / (_max - _min).clamp_min(1e-8)
            return field0.cpu(), gt0.cpu(), pred0.cpu()
    z = torch.zeros((1, 1))
    return z, z, z


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tao2019 D2NN saliency")
    parser.add_argument("--config", required=True, help="spec-style YAML config")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    set_global_seed(int(exp.get("seed", 42)), deterministic=bool(exp.get("deterministic", True)))

    run_dir = create_run_dir(cfg, cwd=Path(__file__).resolve().parents[3])
    save_resolved_config(run_dir / "config_resolved.yaml", cfg)

    device = choose_device(exp)
    log = LiveLogger(
        run_dir=run_dir,
        task="saliency",
        total_epochs=int(cfg["training"]["epochs"]),
        log_interval_steps=int(cfg["training"].get("log_interval_steps", 20)),
        use_color=bool(cfg["training"].get("color_logs", True)),
        show_cuda_memory=bool(cfg["training"].get("show_cuda_memory", True)),
    )
    log.start(experiment_name=str(exp["name"]), device=str(device))

    model = build_model(cfg).to(device)
    train_loader, val_loader = build_dataloaders(cfg)
    gamma_flip_enabled = bool(cfg["task"].get("gamma_flip", True))

    # Compute center-crop box: strip zero-padding for f_max evaluation
    _N = int(cfg["optics"]["grid"]["nx"])
    _preprocess = cfg["data"].get("preprocess", {})
    _obj_h = _N
    _obj_w = _N
    if "resize_to" in _preprocess:
        _obj_h = int(_preprocess["resize_to"][0])
        _obj_w = int(_preprocess["resize_to"][1])
    _y0 = (_N - _obj_h) // 2
    _x0 = (_N - _obj_w) // 2
    eval_crop_box: tuple[int, int, int, int] | None = None
    if _obj_h < _N or _obj_w < _N:
        eval_crop_box = (_y0, _y0 + _obj_h, _x0, _x0 + _obj_w)

    best_ckpt_path = run_dir / "checkpoints" / "best.pt"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_best(info: dict) -> None:
        import copy
        tmp_model = build_model(cfg).to(device)
        tmp_model.load_state_dict(info["state_dict"])
        save_checkpoint(best_ckpt_path, model=tmp_model, optimizer=None, epoch=info["epoch"], extra={"best_metric": info["val_fmax"]})

    # Parse loss configuration
    loss_mode = str(cfg["training"].get("loss_mode", "mse"))
    loss_weights = None
    if "loss_weights" in cfg["training"]:
        loss_weights = {
            'bce': float(cfg["training"]["loss_weights"].get("bce", 1.0)),
            'iou': float(cfg["training"]["loss_weights"].get("iou", 2.0)),
            'structure': float(cfg["training"]["loss_weights"].get("structure", 1.0)),
            'center_penalty': float(cfg["training"]["loss_weights"].get("center_penalty", 0.1)),
        }
    loss_normalization = str(cfg["training"].get("loss_normalization", "pred_only"))
    loss_scope = str(cfg["training"].get("loss_scope", "crop"))

    history = train_saliency(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=float(cfg["training"]["lr"]),
        epochs=int(cfg["training"]["epochs"]),
        gamma_flip=gamma_flip_enabled,
        pr_thresholds=int(cfg["eval"].get("pr_thresholds", 256)),
        compute_train_fmax=bool(cfg["training"].get("compute_train_fmax", False)),
        eval_interval_epochs=int(cfg["training"].get("eval_interval_epochs", 5)),
        lr_scheduler=str(cfg["training"].get("lr_scheduler", "none")),
        lr_min=float(cfg["training"].get("lr_min", 1e-5)),
        max_steps_per_epoch=args.max_steps_per_epoch,
        eval_crop_box=eval_crop_box,
        step_callback=log.on_step,
        epoch_callback=log.on_epoch_end,
        best_state_callback=_save_best,
        loss_mode=loss_mode,
        loss_weights=loss_weights,
        loss_normalization=loss_normalization,
        loss_scope=loss_scope,
    )

    final_epoch = int(cfg["training"]["epochs"])
    save_checkpoint(run_dir / "checkpoints" / "final.pt", model=model, optimizer=None, epoch=final_epoch, extra={"history": history})
    best_idx = best_epoch_index(history, metric_name="val_fmax", maximize=True)
    # best.pt already saved during training via best_state_callback
    # If never saved (no val_fmax computed), fall back to final model
    if not best_ckpt_path.exists():
        best_metric = history["val_fmax"][best_idx] if history["val_fmax"] else None
        save_checkpoint(best_ckpt_path, model=model, optimizer=None, epoch=best_idx + 1, extra={"best_metric": best_metric})

    print("[EVAL] computing final PR/F-max on validation set...", flush=True)
    preds, gts = _collect_preds(
        model,
        val_loader,
        device,
        eval_crop_box=eval_crop_box,
        gamma_flip=gamma_flip_enabled,
    )
    thresholds = int(cfg["eval"].get("pr_thresholds", 256))
    precision, recall = pr_curve(preds, gts, thresholds=thresholds)
    fmax = max_f_measure(preds, gts, thresholds=thresholds, beta2=float(cfg["eval"].get("f_beta2", 0.3)))

    metrics: dict[str, Any] = summarize_history(history)
    metrics["eval_fmax"] = fmax
    metrics["best_epoch"] = best_idx + 1
    save_metrics(run_dir / "metrics.json", metrics)

    field0, gt0, pred0 = _sample_triplet(model, val_loader, device, eval_crop_box=eval_crop_box)
    factory = FigureFactory(run_dir / "figures")
    factory.plot_convergence(history, left_key="val_loss", right_key="val_fmax", name="convergence_saliency.png")
    if field0.numel() > 1:
        if gamma_flip_enabled:
            pred0 = gamma_flip2d(pred0.unsqueeze(0))[0]
        crop_size = min(_obj_h, _obj_w) if eval_crop_box is not None else None
        threshold = float(cfg["eval"].get("viz_threshold", 0.5))
        factory.plot_saliency_grid(inp=field0.real.numpy(), gt=gt0.numpy(), pred=pred0.numpy(), name="saliency_grid.png")
        factory.plot_saliency_diagnostics(
            inp=field0.real.numpy(),
            gt=gt0.numpy(),
            pred=pred0.numpy(),
            crop_size=crop_size,
            threshold=threshold,
            name="saliency_diagnostics.png",
        )
    phases = [layer.phase().detach().cpu().numpy() for layer in model.layers]
    factory.plot_phase_masks(phases, phase_max=float(cfg["model"]["modulation"]["phase_max_rad"]), name="phase_masks.png")
    factory.plot_pr_curve(precision=precision, recall=recall, max_f=fmax, name="pr_curve.png")

    log.finish(run_dir=run_dir)
    print(str(run_dir))


if __name__ == "__main__":
    main()

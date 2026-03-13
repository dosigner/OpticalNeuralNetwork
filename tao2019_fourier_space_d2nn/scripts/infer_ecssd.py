"""Run saliency inference on ECSSD and export visual diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Ensure src is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tao2019_fd2nn.cli.common import build_model, load_config
from tao2019_fd2nn.data.saliency_pairs import SaliencyPairsDataset
from tao2019_fd2nn.optics.fft2c import gamma_flip2d
from tao2019_fd2nn.training.metrics_saliency import max_f_measure, pr_curve
from tao2019_fd2nn.utils.math import intensity


def _resolve_path(base: Path, p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else base / pp


def _center_crop(arr: np.ndarray, size: int | None) -> np.ndarray:
    if size is None:
        return arr
    h, w = arr.shape
    if size <= 0 or size >= min(h, w):
        return arr
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return arr[y0 : y0 + size, x0 : x0 + size]


def _crop_box_and_size(cfg: dict, n: int) -> tuple[tuple[int, int, int, int] | None, int | None]:
    preprocess = cfg.get("data", {}).get("preprocess", {})
    resize_to = preprocess.get("resize_to")
    if not resize_to:
        return None, None
    h = int(resize_to[0])
    w = int(resize_to[1])
    if h >= n and w >= n:
        return None, None
    y0 = (n - h) // 2
    x0 = (n - w) // 2
    return (y0, y0 + h, x0, x0 + w), min(h, w)


def _latest_run_dir(run_base: Path) -> Path:
    run_dirs = sorted([p for p in run_base.iterdir() if p.is_dir()])
    if not run_dirs:
        raise RuntimeError(f"no run directory found under {run_base}")
    return run_dirs[-1]


def _save_gray(arr: np.ndarray, path: Path) -> None:
    img = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def _save_comparison_figure(
    inputs: list[np.ndarray],
    gts: list[np.ndarray],
    preds: list[np.ndarray],
    names: list[str],
    *,
    threshold: float,
    crop_size: int | None,
    path: Path,
    n: int,
) -> None:
    n = min(n, len(inputs))
    fig, axes = plt.subplots(4, n, figsize=(max(6, n * 1.25), 6.2))
    if n == 1:
        axes = np.array(axes).reshape(4, 1)
    row_labels = ["Input (crop)", "GT (crop)", "Pred (crop)", f"Pred>= {threshold:.2f}"]

    for col in range(n):
        inp_c = _center_crop(inputs[col], crop_size)
        gt_c = _center_crop(gts[col], crop_size)
        pred_c = _center_crop(preds[col], crop_size)
        pred_bin = (pred_c >= threshold).astype(np.float32)

        axes[0, col].imshow(inp_c, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col].imshow(gt_c, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, col].imshow(pred_c, cmap="gray", vmin=0.0, vmax=1.0)
        axes[3, col].imshow(pred_bin, cmap="gray", vmin=0.0, vmax=1.0)

        axes[0, col].set_title(names[col], fontsize=6, rotation=45, ha="right")
        for row in range(4):
            axes[row, col].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=6)

    fig.suptitle("F-D2NN Saliency on ECSSD (center-crop + threshold)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_pr_curve(precision: np.ndarray, recall: np.ndarray, max_f: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(recall, precision, color="#d62728", linewidth=2, label=f"Fmax: {max_f:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve on ECSSD")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ECSSD saliency inference")
    parser.add_argument(
        "--config",
        default="src/tao2019_fd2nn/config/saliency_ecssd_f2mm.yaml",
        help="spec-style YAML config",
    )
    parser.add_argument("--run-dir", default=None, help="run directory containing checkpoints/")
    parser.add_argument("--ckpt-name", default="best.pt", help="checkpoint filename under checkpoints/")
    parser.add_argument("--out-dir", default=None, help="output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold for binary mask export")
    parser.add_argument("--samples-figure", type=int, default=20, help="number of samples shown in comparison figure")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(_resolve_path(project_root, args.config))

    exp_name = str(cfg["experiment"]["name"])
    save_dir = str(cfg["experiment"].get("save_dir", "runs"))

    if args.run_dir is None:
        run_base = _resolve_path(project_root, save_dir) / exp_name
        run_dir = _latest_run_dir(run_base)
    else:
        run_dir = _resolve_path(project_root, args.run_dir)

    ckpt_path = run_dir / "checkpoints" / args.ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    out_dir = _resolve_path(
        project_root,
        args.out_dir or (project_root / "inference_results" / f"{exp_name}_ecssd_eval"),
    )
    (out_dir / "input").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_bin").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n = int(cfg["optics"]["grid"]["nx"])
    preprocess = cfg.get("data", {}).get("preprocess", {})
    object_size = int(preprocess.get("resize_to", [n, n])[0])
    image_dir = _resolve_path(project_root, cfg["data"]["val_image_dir"])
    mask_dir = _resolve_path(project_root, cfg["data"]["val_mask_dir"])

    val_ds = SaliencyPairsDataset(image_dir=image_dir, mask_dir=mask_dir, N=n, object_size=object_size)
    eval_crop_box, crop_size = _crop_box_and_size(cfg, n)
    gamma_flip_enabled = bool(cfg["task"].get("gamma_flip", True))

    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print(f"Dataset size: {len(val_ds)}")
    print(f"Gamma flip enabled: {gamma_flip_enabled}")
    print(f"Eval center-crop box: {eval_crop_box}")
    print(f"Output directory: {out_dir}")

    inputs: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    names: list[str] = []
    all_preds: list[torch.Tensor] = []
    all_gts: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            field, target = val_ds[i]
            field_batch = field.unsqueeze(0).to(device)
            out_field = model(field_batch)
            out_i = intensity(out_field)
            i_min = out_i.amin(dim=(-2, -1), keepdim=True)
            i_max = out_i.amax(dim=(-2, -1), keepdim=True)
            pred_tensor = (out_i - i_min) / (i_max - i_min).clamp_min(1e-8)

            pred_cpu = pred_tensor.cpu()
            gt_cpu = target.unsqueeze(0).cpu()
            if gamma_flip_enabled:
                pred_cpu = gamma_flip2d(pred_cpu)
            if eval_crop_box is not None:
                y0, y1, x0, x1 = eval_crop_box
                pred_metric = pred_cpu[..., y0:y1, x0:x1]
                gt_metric = gt_cpu[..., y0:y1, x0:x1]
            else:
                pred_metric = pred_cpu
                gt_metric = gt_cpu

            all_preds.append(pred_metric)
            all_gts.append(gt_metric)

            pred = pred_cpu[0].numpy()
            inp = field.real.numpy()
            gt = target.numpy()
            pred_bin = (pred >= float(args.threshold)).astype(np.float32)

            name = val_ds.image_files[i].stem
            names.append(name)
            inputs.append(inp)
            gts.append(gt)
            preds.append(pred)

            _save_gray(inp, out_dir / "input" / f"{name}.png")
            _save_gray(gt, out_dir / "gt" / f"{name}.png")
            _save_gray(pred, out_dir / "pred" / f"{name}.png")
            _save_gray(pred_bin, out_dir / "pred_bin" / f"{name}.png")

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i + 1}/{len(val_ds)}] processed {name}")

    preds_tensor = torch.cat(all_preds, dim=0)
    gts_tensor = torch.cat(all_gts, dim=0)
    thresholds = int(cfg.get("eval", {}).get("pr_thresholds", 256))
    beta2 = float(cfg.get("eval", {}).get("f_beta2", 0.3))
    precision, recall = pr_curve(preds_tensor, gts_tensor, thresholds=thresholds)
    fmax = max_f_measure(preds_tensor, gts_tensor, thresholds=thresholds, beta2=beta2)

    metrics = {
        "config": str(_resolve_path(project_root, args.config)),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "num_images": int(len(val_ds)),
        "threshold": float(args.threshold),
        "pr_thresholds": thresholds,
        "f_beta2": beta2,
        "gamma_flip": gamma_flip_enabled,
        "eval_crop_box": list(eval_crop_box) if eval_crop_box is not None else None,
        "fmax": float(fmax),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _save_comparison_figure(
        inputs,
        gts,
        preds,
        names,
        threshold=float(args.threshold),
        crop_size=crop_size,
        path=out_dir / "comparison_figure.png",
        n=int(args.samples_figure),
    )
    _save_pr_curve(precision, recall, fmax, out_dir / "pr_curve.png")

    print(f"Fmax on ECSSD: {fmax:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()

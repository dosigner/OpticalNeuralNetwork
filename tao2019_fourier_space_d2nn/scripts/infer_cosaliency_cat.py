"""Run inference on CIFAR-10 cat RBD saliency model and save results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tao2019_fd2nn.cli.common import build_model, load_config
from tao2019_fd2nn.data.saliency_pairs import SaliencyPairsDataset
from tao2019_fd2nn.utils.math import intensity


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "src" / "tao2019_fd2nn" / "config" / "saliency_cifar_cosaliency_cat.yaml"
    cfg = load_config(str(config_path))

    # Find the latest checkpoint
    run_base = project_root / "runs" / "saliency_cifar_cosaliency_cat"
    run_dirs = sorted(run_base.iterdir()) if run_base.exists() else []
    if not run_dirs:
        print("ERROR: no run directory found under", run_base)
        return
    latest_run = run_dirs[-1]
    ckpt_path = latest_run / "checkpoints" / "best.pt"
    print(f"Loading checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load weights
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build dataset (test: horse category — paper uses Horse as cross-class test)
    val_ds = SaliencyPairsDataset(
        image_dir=str(project_root / "data" / "cifar10_object_rbd_all" / "val" / "images" / "horse"),
        mask_dir=str(project_root / "data" / "cifar10_object_rbd_all" / "val" / "targets" / "horse"),
        N=160,
        object_size=100,
    )

    # Output directory
    out_dir = project_root / "inference_results" / "rbd_horse"
    (out_dir / "input").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred").mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(val_ds)} results to: {out_dir}")

    inputs, gts, preds, names = [], [], [], []

    with torch.no_grad():
        for i in range(len(val_ds)):
            field, target = val_ds[i]
            field_batch = field.unsqueeze(0).to(device)
            out_field = model(field_batch)
            _I = intensity(out_field)
            _min = _I.amin(dim=(-2, -1), keepdim=True)
            _max = _I.amax(dim=(-2, -1), keepdim=True)
            pred = ((_I - _min) / (_max - _min).clamp_min(1e-8))[0].cpu().numpy()
            inp = field.real.numpy()
            gt = target.numpy()

            name = val_ds.image_files[i].stem
            names.append(name)
            inputs.append(inp)
            gts.append(gt)
            preds.append(pred)

            _save_gray(inp, out_dir / "input" / f"{name}.png")
            _save_gray(gt, out_dir / "gt" / f"{name}.png")
            _save_gray(pred, out_dir / "pred" / f"{name}.png")

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{len(val_ds)}] {name}")

    print(f"Done! Results saved to: {out_dir}")

    # Save comparison figure (first N samples)
    _save_comparison_figure(inputs, gts, preds, names, out_dir / "comparison_figure.png", n=20)
    print(f"Comparison figure saved to: {out_dir / 'comparison_figure.png'}")


def _save_gray(arr: np.ndarray, path: Path) -> None:
    img = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def _save_comparison_figure(
    inputs: list,
    gts: list,
    preds: list,
    names: list,
    path: Path,
    n: int = 20,
) -> None:
    n = min(n, len(inputs))
    # Crop center 100x100 (remove padding) for display
    def crop(arr):
        h, w = arr.shape
        cy, cx = h // 2, w // 2
        r = 50
        return arr[cy - r : cy + r, cx - r : cx + r]

    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5))
    row_labels = ["Input", "Target (RBD)", "Prediction"]

    for col in range(n):
        axes[0, col].imshow(crop(inputs[col]), cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(crop(gts[col]), cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(crop(preds[col]), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(names[col], fontsize=5, rotation=45, ha="right")
        for row in range(3):
            axes[row, col].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=5)
        axes[row, 0].yaxis.set_label_position("left")
        axes[row, 0].tick_params(left=False, bottom=False)
        axes[row, 0].set_yticks([])

    fig.suptitle("F-D2NN Saliency (trained: Cat / tested: Horse): Input / RBD Target / Prediction", fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

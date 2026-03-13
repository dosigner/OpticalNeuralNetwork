"""Generate a detailed comparison figure from saved ECSSD inference results."""

from __future__ import annotations

import sys
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parents[1]
out_dir = project_root / "inference_results" / "ecssd_eval"
fig_dir = out_dir

input_dir = out_dir / "input"
gt_dir = out_dir / "gt"
pred_dir = out_dir / "pred"


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L")).astype(np.float32) / 255.0


def crop_center(arr: np.ndarray, size: int = 100) -> np.ndarray:
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    r = size // 2
    return arr[cy - r: cy + r, cx - r: cx + r]


def main() -> None:
    # Pick samples: first 8 + 8 random from the rest
    all_names = sorted([p.stem for p in input_dir.glob("*.png")])
    n_show = min(16, len(all_names))
    first_n = all_names[:4]
    rest = all_names[4:]
    random.seed(42)
    sampled = first_n + random.sample(rest, min(n_show - 4, len(rest)))
    sampled = sampled[:n_show]

    rows = 3  # input, target, pred
    cols = len(sampled)
    row_labels = ["Input", "Ground Truth", "Prediction"]
    row_cmaps = ["gray", "gray", "hot"]

    fig = plt.figure(figsize=(cols * 1.6 + 0.8, rows * 1.8 + 0.8))
    fig.patch.set_facecolor("#111111")

    gs = gridspec.GridSpec(
        rows, cols,
        figure=fig,
        left=0.07, right=0.99,
        top=0.90, bottom=0.03,
        hspace=0.05, wspace=0.03,
    )

    for col, name in enumerate(sampled):
        inp = crop_center(load_gray(input_dir / f"{name}.png"))
        gt = crop_center(load_gray(gt_dir / f"{name}.png"))
        pred = crop_center(load_gray(pred_dir / f"{name}.png"))
        data = [inp, gt, pred]

        for row in range(rows):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(data[row], cmap=row_cmaps[row], vmin=0, vmax=1, interpolation="bilinear")
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(
                    row_labels[row],
                    fontsize=9, color="white", fontweight="bold",
                    rotation=90, labelpad=4,
                    va="center",
                )
                ax.yaxis.set_label_position("left")
                ax.tick_params(left=False, bottom=False)
                ax.set_yticks([])

            if row == 0:
                ax.set_title(name, fontsize=7, color="#aaaaaa", pad=2)

    fig.suptitle(
        "F-D²NN Saliency Detection on ECSSD (Input / Ground Truth / Prediction)",
        fontsize=13, color="white", fontweight="bold", y=0.97,
    )

    out_path = fig_dir / "comparison_figure_detail.png"
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure: {out_path}")


if __name__ == "__main__":
    main()

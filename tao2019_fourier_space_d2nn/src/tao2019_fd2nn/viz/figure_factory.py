"""Figure factory with both generic and pixel-layout rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from tao2019_fd2nn.viz.d2nn_schematic import draw_d2nn_config_schematic
from tao2019_fd2nn.viz.layout_specs import LayoutSpec, get_layout, make_axes_from_layout


class FigureFactory:
    """Centralized plotting utility for all experiment outputs."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.out_dir / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _layout_canvas(self, layout_id: str) -> tuple[LayoutSpec, plt.Figure, dict[str, object]]:
        layout = get_layout(layout_id)
        fig = plt.figure(figsize=(layout.width_px / layout.dpi, layout.height_px / layout.dpi), dpi=layout.dpi)
        axd = make_axes_from_layout(fig, layout)
        return layout, fig, axd

    def _render_layout_texts(self, fig: plt.Figure, layout: LayoutSpec) -> None:
        W, H = layout.width_px, layout.height_px
        for t in layout.texts.values():
            fig.text(
                float(t["x"]) / W,
                1.0 - float(t["y"]) / H,
                str(t["s"]),
                ha=str(t.get("ha", "center")),
                va=str(t.get("va", "center")),
                rotation=float(t.get("rotation", 0.0)),
            )

    def render_fig2(self, rows: list[list[np.ndarray]], phase_row: list[np.ndarray], *, name: str = "figure_fig2.png") -> Path:
        """Render Figure 2 layout.

        rows: 3x5 image arrays (target/co-saliency/output rows)
        phase_row: 5 phase maps for row 4
        """

        layout, fig, axd = self._layout_canvas("fig2")
        for r in range(3):
            for c in range(5):
                ax = axd[f"img_r{r}_c{c}"]
                im = ax.imshow(rows[r][c], cmap="gray", origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 4:
                    cax = axd[f"cbar_r{r}"]
                    fig.colorbar(im, cax=cax)

        for c in range(5):
            ax = axd[f"img_r3_c{c}"]
            im = ax.imshow(phase_row[c], cmap="twilight", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 4:
                cax = axd["cbar_phase"]
                fig.colorbar(im, cax=cax)

        self._render_layout_texts(fig, layout)
        return self._save(fig, name)

    def render_fig3(self, rows: list[list[np.ndarray]], phase_row: list[np.ndarray], *, name: str = "figure_fig3.png") -> Path:
        """Render Figure 3 layout (same panel geometry as fig2)."""

        layout, fig, axd = self._layout_canvas("fig3")
        for r in range(3):
            for c in range(5):
                ax = axd[f"img_r{r}_c{c}"]
                im = ax.imshow(rows[r][c], cmap="gray", origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 4:
                    cax = axd[f"cbar_r{r}"]
                    fig.colorbar(im, cax=cax)

        for c in range(5):
            ax = axd[f"img_r3_c{c}"]
            im = ax.imshow(phase_row[c], cmap="twilight", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 4:
                cax = axd["cbar_phase"]
                fig.colorbar(im, cax=cax)

        self._render_layout_texts(fig, layout)
        return self._save(fig, name)

    def render_fig4(
        self,
        a_curves: dict[str, list[float]],
        b_curves: dict[str, list[float]],
        *,
        name: str = "figure_fig4.png",
    ) -> Path:
        """Render Figure 4 layout from curve histories."""

        layout, fig, axd = self._layout_canvas("fig4")
        for i in range(4):
            axd[f"a_cfg{i}"].axis("off")
            axd[f"b_cfg{i}"].axis("off")
        for r in range(4):
            for c in range(3):
                axd[f"a_img_r{r}_c{c}"].axis("off")

        ax_a = axd["a_plot"]
        for k, v in a_curves.items():
            ax_a.plot(v, label=k)
        ax_a.set_title("Convergence (a)")
        ax_a.set_xlabel("Epoch")
        ax_a.set_ylabel("Accuracy")
        ax_a.grid(alpha=0.2)
        ax_a.legend()

        ax_b = axd["b_plot"]
        for k, v in b_curves.items():
            ax_b.plot(v, label=k)
        ax_b.set_title("Convergence (b)")
        ax_b.set_xlabel("Epoch")
        ax_b.set_ylabel("Accuracy")
        ax_b.grid(alpha=0.2)
        ax_b.legend()

        self._render_layout_texts(fig, layout)
        return self._save(fig, name)

    def plot_mnist_fig4a_comparison(
        self,
        curves: dict[str, list[float]],
        max_acc: dict[str, float],
        *,
        name: str = "fig4_mnist_5l_epoch30_bs10.png",
        legend_title: str | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> Path:
        """Render MNIST 5-layer Fig.4a style accuracy comparison."""

        ordered_labels = [
            "Linear Real",
            "Nonlinear Real",
            "Linear Fourier",
            "Nonlinear Fourier",
        ]
        colors = {
            "Linear Real": "#1f77b4",
            "Nonlinear Real": "#d95319",
            "Linear Fourier": "#edb120",
            "Nonlinear Fourier": "#7e2f8e",
        }
        missing = [label for label in ordered_labels if label not in curves]
        if missing:
            raise ValueError(f"missing curves for labels: {', '.join(missing)}")

        fig, ax = plt.subplots(figsize=(7.28, 5.0), dpi=100)
        fig.patch.set_facecolor("#e8e8e8")
        ax.set_facecolor("#efefef")

        max_len = 0
        for label in ordered_labels:
            values = [float(v) for v in curves[label]]
            if not values:
                raise ValueError(f"curve is empty for label: {label}")
            max_len = max(max_len, len(values))
            curve_max = float(max_acc.get(label, max(values)))
            y = np.array([0.0, *values], dtype=float)
            x = np.arange(0, len(values) + 1)
            ax.plot(x, y, color=colors[label], lw=2.0, label=f"{label} ({curve_max * 100.0:.1f}%)")

        x_max = max(30, max_len)
        ax.set_title("MNIST Dataset Classification", fontsize=24)
        ax.set_xlabel("Epoch Number", fontsize=24)
        ax.set_ylabel("Classification Accuracy", fontsize=22)
        ax.set_xlim(0, x_max)
        ylim_val = ylim or (0.8, 1.0)
        ax.set_ylim(*ylim_val)
        if x_max == 30:
            ax.set_xticks([0, 10, 20, 30])
        else:
            ticks = [t for t in range(0, x_max + 1, 10)]
            if ticks[-1] != x_max:
                ticks.append(x_max)
            ax.set_xticks(ticks)
        ax.set_yticks([round(v, 1) for v in np.arange(ylim_val[0], ylim_val[1] + 0.001, 0.1)])
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=20, width=1.7)
        for spine in ax.spines.values():
            spine.set_linewidth(1.7)

        legend = ax.legend(
            loc="lower right",
            fontsize=17,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="#1f1f1f",
            title=legend_title or "5 Layers D$^2$NN,\nMaximum Validation Accuracy",
            title_fontsize=19,
        )
        legend._legend_box.align = "left"
        fig.tight_layout()
        return self._save(fig, name)

    @staticmethod
    def _draw_accuracy_curves(
        ax: plt.Axes,
        curves: dict[str, list[float]],
        max_acc: dict[str, float],
        *,
        ordered_labels: list[str],
        colors: dict[str, str],
        legend_title: str | None = None,
        ylim: tuple[float, float] | None = None,
        title_fontsize: int = 24,
        label_fontsize: int = 22,
        tick_fontsize: int = 20,
        legend_fontsize: int = 17,
        legend_title_fontsize: int = 19,
    ) -> None:
        """Draw accuracy-vs-epoch curves on the given axes (shared helper)."""
        ax.set_facecolor("#efefef")
        max_len = 0
        for label in ordered_labels:
            values = [float(v) for v in curves[label]]
            if not values:
                raise ValueError(f"curve is empty for label: {label}")
            max_len = max(max_len, len(values))
            curve_max = float(max_acc.get(label, max(values)))
            y = np.array([0.0, *values], dtype=float)
            x = np.arange(0, len(values) + 1)
            ax.plot(x, y, color=colors[label], lw=2.0, label=f"{label} ({curve_max * 100.0:.1f}%)")

        x_max = max(30, max_len)
        ax.set_title("MNIST Dataset Classification", fontsize=title_fontsize)
        ax.set_xlabel("Epoch Number", fontsize=label_fontsize)
        ax.set_ylabel("Classification Accuracy", fontsize=label_fontsize)
        ax.set_xlim(0, x_max)
        ylim_val = ylim or (0.8, 1.0)
        ax.set_ylim(*ylim_val)
        if x_max == 30:
            ax.set_xticks([0, 10, 20, 30])
        else:
            ticks = [t for t in range(0, x_max + 1, 10)]
            if ticks[-1] != x_max:
                ticks.append(x_max)
            ax.set_xticks(ticks)
        ax.set_yticks([round(v, 1) for v in np.arange(ylim_val[0], ylim_val[1] + 0.001, 0.1)])
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=tick_fontsize, width=1.7)
        for spine in ax.spines.values():
            spine.set_linewidth(1.7)

        legend = ax.legend(
            loc="lower right",
            fontsize=legend_fontsize,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="#1f1f1f",
            title=legend_title or "5 Layers D$^2$NN,\nMaximum Testing Accuracy",
            title_fontsize=legend_title_fontsize,
        )
        legend._legend_box.align = "left"

    def plot_fig4a_with_schematics(
        self,
        curves: dict[str, list[float]],
        max_acc: dict[str, float],
        *,
        name: str = "fig4a_mnist_5l_with_schematics.png",
        legend_title: str | None = None,
        ylim: tuple[float, float] | None = None,
        timing_info: dict[str, float] | None = None,
    ) -> Path:
        """Render Fig.4a with config schematics (top) + accuracy curves (bottom)."""

        ordered_labels = ["Linear Real", "Nonlinear Real", "Linear Fourier", "Nonlinear Fourier"]
        config_types = ["linear_real", "nonlinear_real", "linear_fourier", "nonlinear_fourier"]
        colors = {
            "Linear Real": "#1f77b4",
            "Nonlinear Real": "#d95319",
            "Linear Fourier": "#edb120",
            "Nonlinear Fourier": "#7e2f8e",
        }
        missing = [label for label in ordered_labels if label not in curves]
        if missing:
            raise ValueError(f"missing curves for labels: {', '.join(missing)}")

        fig = plt.figure(figsize=(14.5, 10.0), dpi=150)
        fig.patch.set_facecolor("#e8e8e8")
        gs = gridspec.GridSpec(2, 4, height_ratios=[1, 2.5], hspace=0.25, wspace=0.12)

        # Top row: 4 schematics
        for i, (cfg_type, label) in enumerate(zip(config_types, ordered_labels)):
            ax_cfg = fig.add_subplot(gs[0, i])
            draw_d2nn_config_schematic(ax_cfg, cfg_type)

        # Bottom: accuracy curves (spans all 4 columns)
        ax_plot = fig.add_subplot(gs[1, :])
        self._draw_accuracy_curves(
            ax_plot, curves, max_acc,
            ordered_labels=ordered_labels, colors=colors,
            legend_title=legend_title, ylim=ylim,
        )

        # Timing annotation
        if timing_info:
            lines = [f"{lbl}: {timing_info[lbl]/60:.1f} min" for lbl in ordered_labels if lbl in timing_info]
            if lines:
                timing_text = "Training Time\n" + "\n".join(lines)
                fig.text(0.02, 0.02, timing_text, fontsize=8, va="bottom", ha="left",
                         family="monospace", color="#555555")

        return self._save(fig, name)

    def plot_comparison(
        self,
        curves: dict[str, list[float]],
        max_acc: dict[str, float],
        *,
        ordered_labels: list[str],
        colors: dict[str, str],
        name: str = "comparison.png",
        legend_title: str | None = None,
        ylim: tuple[float, float] | None = None,
        title: str = "MNIST Dataset Classification",
    ) -> Path:
        """Generic accuracy-vs-epoch comparison plot."""

        missing = [label for label in ordered_labels if label not in curves]
        if missing:
            raise ValueError(f"missing curves for labels: {', '.join(missing)}")

        fig, ax = plt.subplots(figsize=(7.28, 5.0), dpi=100)
        fig.patch.set_facecolor("#e8e8e8")
        ax.set_facecolor("#efefef")

        max_len = 0
        for label in ordered_labels:
            values = [float(v) for v in curves[label]]
            if not values:
                raise ValueError(f"curve is empty for label: {label}")
            max_len = max(max_len, len(values))
            curve_max = float(max_acc.get(label, max(values)))
            y = np.array([0.0, *values], dtype=float)
            x = np.arange(0, len(values) + 1)
            ax.plot(x, y, color=colors[label], lw=2.0, label=f"{label} ({curve_max * 100.0:.1f}%)")

        x_max = max(30, max_len)
        ax.set_title(title, fontsize=24)
        ax.set_xlabel("Epoch Number", fontsize=24)
        ax.set_ylabel("Classification Accuracy", fontsize=22)
        ax.set_xlim(0, x_max)
        ylim_val = ylim or (0.8, 1.0)
        ax.set_ylim(*ylim_val)
        if x_max == 30:
            ax.set_xticks([0, 10, 20, 30])
        else:
            ticks = [t for t in range(0, x_max + 1, 10)]
            if ticks[-1] != x_max:
                ticks.append(x_max)
            ax.set_xticks(ticks)
        ax.set_yticks([round(v, 1) for v in np.arange(ylim_val[0], ylim_val[1] + 0.001, 0.1)])
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=20, width=1.7)
        for spine in ax.spines.values():
            spine.set_linewidth(1.7)

        legend = ax.legend(
            loc="lower right",
            fontsize=14,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="#1f1f1f",
            title=legend_title or "Maximum Validation Accuracy",
            title_fontsize=16,
        )
        legend._legend_box.align = "left"
        fig.tight_layout()
        return self._save(fig, name)

    def plot_saliency_grid(self, inp: np.ndarray, gt: np.ndarray, pred: np.ndarray, *, name: str = "saliency_grid.png") -> Path:
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
        axes[0].imshow(inp, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        axes[0].set_title("Input")
        axes[1].imshow(gt, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        axes[1].set_title("GT")
        axes[2].imshow(pred, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        axes[2].set_title("Pred")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return self._save(fig, name)

    def plot_phase_masks(self, phases: Iterable[np.ndarray], *, phase_max: float, name: str = "phase_masks.png") -> Path:
        phase_list = list(phases)
        n = len(phase_list)
        cols = min(5, max(1, n))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.8 * rows))
        axes_a = np.array(axes).reshape(-1)
        for i, phase in enumerate(phase_list):
            im = axes_a[i].imshow(phase, cmap="twilight", origin="lower", vmin=0.0, vmax=float(phase_max))
            axes_a[i].set_title(f"Layer {i+1}")
            axes_a[i].set_xticks([])
            axes_a[i].set_yticks([])
            fig.colorbar(im, ax=axes_a[i], shrink=0.75)
        for j in range(i + 1, len(axes_a)):
            axes_a[j].axis("off")
        fig.tight_layout()
        return self._save(fig, name)

    def plot_pr_curve(self, precision: np.ndarray, recall: np.ndarray, *, max_f: float | None = None, name: str = "pr_curve.png") -> Path:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        ax.plot(recall, precision, lw=2.0)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        title = "PR Curve"
        if max_f is not None:
            title += f" (Fmax={max_f:.3f})"
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, name)

    def plot_convergence(self, history: dict[str, list[float]], *, left_key: str, right_key: str | None = None, name: str = "convergence.png") -> Path:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(history.get(left_key, []), label=left_key, color="#1f4cff")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(left_key)
        if right_key:
            ax2 = ax.twinx()
            ax2.plot(history.get(right_key, []), label=right_key, color="#f54242")
            ax2.set_ylabel(right_key)
        ax.grid(alpha=0.2)
        ax.set_title("Convergence")
        fig.tight_layout()
        return self._save(fig, name)

    def plot_sensitivity(self, x: np.ndarray, ys: dict[str, np.ndarray], *, xlabel: str, ylabel: str, name: str = "sensitivity.png") -> Path:
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        for label, values in ys.items():
            ax.plot(x, values, marker="o", lw=1.6, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return self._save(fig, name)

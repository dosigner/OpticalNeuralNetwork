"""Figure factory with both generic and pixel-layout rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from tao2019_fd2nn.viz.d2nn_schematic import draw_d2nn_config_schematic, draw_s7_paper_inset
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

    @staticmethod
    def _s8_style(label: str) -> dict[str, object]:
        lower = label.lower()
        if "nonlinear fourier" in lower:
            color = "#7e2f8e"
        elif "nonlinear real" in lower:
            color = "#d95319"
        elif "linear real" in lower:
            color = "#1f77b4"
        else:
            color = "#4d4d4d"

        if "1l" in lower or "single" in lower:
            marker = "o"
            linestyle = "-"
        elif "2l" in lower or "two" in lower:
            marker = "s"
            linestyle = "--"
        elif "5l" in lower or "five" in lower:
            marker = "D"
            linestyle = "-."
        else:
            marker = "o"
            linestyle = "-"
        return {"color": color, "marker": marker, "linestyle": linestyle}

    def plot_s8_classification_sensitivity(
        self,
        fabrication_curves: dict[str, list[float]],
        alignment_curves: dict[str, list[float]],
        *,
        fabrication_x: list[float],
        alignment_x: list[float],
        name: str = "supp_s8_cls.png",
    ) -> Path:
        """Render Supplementary Figure S8 classification sensitivity panels."""

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=200)
        fig.patch.set_facecolor("#e6e6e6")
        panel_specs = [
            (axes[0], fabrication_curves, fabrication_x, "(a) Fabrication Imprecision", "Gaussian Sigma (px)"),
            (axes[1], alignment_curves, alignment_x, "(b) Alignment Error", "Global Shift (um)"),
        ]

        for ax, curves, xs, title, xlabel in panel_specs:
            ax.set_facecolor("#efefef")
            for label, values in curves.items():
                if len(values) != len(xs):
                    raise ValueError(f"{label} has {len(values)} points, expected {len(xs)}")
                style = self._s8_style(label)
                ax.plot(xs, values, label=label, lw=1.8, ms=5.5, **style)
            ax.set_title(title, fontsize=18, pad=10)
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_ylabel("Classification Accuracy", fontsize=16)
            ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=12, width=1.4, length=4)
            for spine in ax.spines.values():
                spine.set_linewidth(1.4)
            ax.grid(alpha=0.18, linewidth=0.8)
            ax.set_ylim(0.0, 1.0)
            legend = ax.legend(
                loc="lower left",
                fontsize=10,
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                edgecolor="#202020",
            )
            legend._legend_box.align = "left"

        fig.tight_layout()
        return self._save(fig, name)

    @staticmethod
    def _draw_s7cd_panel(
        ax: plt.Axes,
        curves: dict[str, list[float]],
        max_acc: dict[str, float],
        *,
        ordered_labels: list[str],
        colors: dict[str, str],
        title: str,
        panel_label: str,
        ylim: tuple[float, float],
        insets: list[dict[str, object]],
        legend_loc: str = "lower right",
    ) -> None:
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
            ax.plot(x, y, color=colors[label], lw=1.8, label=f"{label} ({curve_max * 100.0:.1f}%)")

        x_max = max(30, max_len)
        ax.set_xlim(0, x_max)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=20, pad=10)
        ax.set_xlabel("Epoch Number", fontsize=18)
        ax.set_ylabel("Accuracy", fontsize=18)
        ax.set_xticks([0, 10, 20, 30] if x_max == 30 else list(range(0, x_max + 1, 10)))
        ax.set_yticks([round(v, 1) for v in np.arange(ylim[0], ylim[1] + 0.001, 0.1)])
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=14, width=1.6, length=5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

        legend = ax.legend(
            loc=legend_loc,
            fontsize=12,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="#202020",
            borderpad=0.4,
            labelspacing=0.3,
            handlelength=2.3,
            handletextpad=0.6,
        )
        legend._legend_box.align = "left"
        ax.text(-0.10, 1.03, panel_label, transform=ax.transAxes, fontsize=24, va="bottom", ha="left")

        for inset in insets:
            inset_ax = ax.inset_axes(inset["bbox"])
            draw_s7_paper_inset(
                inset_ax,
                str(inset["config_key"]),
                num_layers=int(inset["num_layers"]),
                border_color=str(inset["border_color"]),
                label=str(inset["label"]),
            )

    def plot_s7cd_panel(
        self,
        curves: dict[str, list[float]],
        max_acc: dict[str, float],
        *,
        ordered_labels: list[str],
        colors: dict[str, str],
        title: str,
        panel_label: str,
        ylim: tuple[float, float],
        insets: list[dict[str, object]],
        name: str,
        legend_loc: str = "lower right",
    ) -> Path:
        """Render one paper-style panel for Supplementary Fig. S7(c) or (d)."""

        fig, ax = plt.subplots(figsize=(7.1, 5.2), dpi=200)
        fig.patch.set_facecolor("#e6e6e6")
        self._draw_s7cd_panel(
            ax,
            curves,
            max_acc,
            ordered_labels=ordered_labels,
            colors=colors,
            title=title,
            panel_label=panel_label,
            ylim=ylim,
            insets=insets,
            legend_loc=legend_loc,
        )
        fig.tight_layout()
        return self._save(fig, name)

    def plot_s7cd_composite(
        self,
        left_curves: dict[str, list[float]],
        left_max_acc: dict[str, float],
        *,
        left_ordered_labels: list[str],
        left_colors: dict[str, str],
        left_title: str,
        left_panel_label: str,
        left_ylim: tuple[float, float],
        left_insets: list[dict[str, object]],
        right_curves: dict[str, list[float]],
        right_max_acc: dict[str, float],
        right_ordered_labels: list[str],
        right_colors: dict[str, str],
        right_title: str,
        right_panel_label: str,
        right_ylim: tuple[float, float],
        right_insets: list[dict[str, object]],
        name: str = "supp_s7cd_mnist_convergence.png",
    ) -> Path:
        """Render the combined paper-style Supplementary Fig. S7(c)(d) figure."""

        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=200)
        fig.patch.set_facecolor("#e6e6e6")
        self._draw_s7cd_panel(
            axes[0],
            left_curves,
            left_max_acc,
            ordered_labels=left_ordered_labels,
            colors=left_colors,
            title=left_title,
            panel_label=left_panel_label,
            ylim=left_ylim,
            insets=left_insets,
            legend_loc="lower right",
        )
        self._draw_s7cd_panel(
            axes[1],
            right_curves,
            right_max_acc,
            ordered_labels=right_ordered_labels,
            colors=right_colors,
            title=right_title,
            panel_label=right_panel_label,
            ylim=right_ylim,
            insets=right_insets,
            legend_loc="lower right",
        )
        fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.15, wspace=0.15)
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

    @staticmethod
    def _center_crop(arr: np.ndarray, size: int | None) -> np.ndarray:
        if size is None:
            return arr
        h, w = int(arr.shape[0]), int(arr.shape[1])
        crop = int(size)
        if crop <= 0 or crop >= min(h, w):
            return arr
        y0 = (h - crop) // 2
        x0 = (w - crop) // 2
        return arr[y0 : y0 + crop, x0 : x0 + crop]

    def plot_saliency_diagnostics(
        self,
        inp: np.ndarray,
        gt: np.ndarray,
        pred: np.ndarray,
        *,
        crop_size: int | None = None,
        threshold: float = 0.5,
        name: str = "saliency_diagnostics.png",
    ) -> Path:
        pred_mask = (pred >= float(threshold)).astype(np.float32)
        inp_crop = self._center_crop(inp, crop_size)
        gt_crop = self._center_crop(gt, crop_size)
        pred_crop = self._center_crop(pred, crop_size)
        pred_mask_crop = self._center_crop(pred_mask, crop_size)

        fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0))
        row0 = (inp, gt, pred, pred_mask)
        row1 = (inp_crop, gt_crop, pred_crop, pred_mask_crop)
        titles = ("Input", "GT", "Pred", f"Pred>= {threshold:.2f}")
        row_labels = ("Full", "Center Crop")

        for c, title in enumerate(titles):
            axes[0, c].set_title(title)
        for r, row_data in enumerate((row0, row1)):
            for c, img in enumerate(row_data):
                axes[r, c].imshow(img, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
            axes[r, 0].set_ylabel(row_labels[r])

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

    @staticmethod
    def _s8_family_color(label: str) -> str:
        if label.startswith("Nonlinear Fourier"):
            return "#7e2f8e"
        if label.startswith("Nonlinear Real"):
            return "#d95319"
        if label.startswith("Linear Real"):
            return "#1f77b4"
        return "#333333"

    @staticmethod
    def _s8_layer_marker(label: str) -> str:
        if "(1L)" in label:
            return "o"
        if "(2L)" in label:
            return "s"
        if "(5L)" in label:
            return "^"
        return "o"

    @classmethod
    def _plot_s8_panel(
        cls,
        ax: plt.Axes,
        *,
        curves: dict[str, list[float]],
        x: list[float] | np.ndarray,
        xlabel: str,
        panel_label: str,
    ) -> None:
        for label, values in curves.items():
            ax.plot(
                x,
                values,
                color=cls._s8_family_color(label),
                marker=cls._s8_layer_marker(label),
                lw=1.8,
                markersize=5,
                label=label,
            )
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("Classification Accuracy", fontsize=15)
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=12, width=1.4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
        ax.grid(alpha=0.25)
        ax.text(-0.12, 1.03, panel_label, transform=ax.transAxes, fontsize=18, va="bottom", ha="left")

    def plot_s8_sensitivity(
        self,
        *,
        fabrication_curves: dict[str, list[float]],
        fabrication_x: list[float] | np.ndarray,
        alignment_curves: dict[str, list[float]],
        alignment_x: list[float] | np.ndarray,
        name: str = "supp_s8.png",
    ) -> Path:
        """Render Supplementary Figure S8 classification sensitivity panels."""

        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=200)
        fig.patch.set_facecolor("#f2f2f2")
        for ax in axes:
            ax.set_facecolor("white")

        self._plot_s8_panel(
            axes[0],
            curves=fabrication_curves,
            x=fabrication_x,
            xlabel="Fabrication Imprecision, Gaussian sigma (px)",
            panel_label="(a)",
        )
        self._plot_s8_panel(
            axes[1],
            curves=alignment_curves,
            x=alignment_x,
            xlabel="Alignment Error (um)",
            panel_label="(b)",
        )
        axes[0].set_title("Fabrication Imprecision", fontsize=17)
        axes[1].set_title("Alignment Error", fontsize=17)
        legend = axes[1].legend(
            loc="lower left",
            fontsize=9,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="#1f1f1f",
            ncol=1,
        )
        legend._legend_box.align = "left"
        fig.tight_layout()
        return self._save(fig, name)

    def plot_accuracy_vs_layers(
        self,
        data: dict[str, dict[int, float]],
        *,
        ordered_labels: list[str],
        colors: dict[str, str],
        markers: dict[str, str],
        schematic_keys: list[str] | None = None,
        schematic_num_layers: int | list[int] = 5,
        name: str = "s7a_accuracy_vs_layers.png",
        ylim: tuple[float, float] = (0.6, 1.0),
    ) -> Path:
        """S7(a)-style plot: accuracy vs layer number with optional schematic insets."""
        from tao2019_fd2nn.viz.d2nn_schematic import draw_s7_schematic

        has_schematics = schematic_keys is not None and len(schematic_keys) > 0
        if isinstance(schematic_num_layers, int):
            schematic_layers = [int(schematic_num_layers)] * (len(schematic_keys) if schematic_keys is not None else 0)
        else:
            schematic_layers = [int(v) for v in schematic_num_layers]
        if has_schematics and len(schematic_layers) != len(schematic_keys):
            raise ValueError("schematic_num_layers must match schematic_keys length")
        if has_schematics:
            fig = plt.figure(figsize=(8.0, 7.0), dpi=150)
            fig.patch.set_facecolor("white")
            gs = gridspec.GridSpec(
                2, len(schematic_keys),
                height_ratios=[1, 2.5], hspace=0.15, wspace=0.08,
                left=0.10, right=0.95, top=0.95, bottom=0.08,
            )
            for i, sk in enumerate(schematic_keys):
                ax_s = fig.add_subplot(gs[0, i])
                draw_s7_schematic(ax_s, sk, num_layers=schematic_layers[i])
            ax = fig.add_subplot(gs[1, :])
        else:
            fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
            fig.patch.set_facecolor("white")

        ax.set_facecolor("white")
        all_layers: set[int] = set()
        for label in ordered_labels:
            acc_by_layer = data[label]
            layers = sorted(acc_by_layer.keys())
            all_layers.update(int(v) for v in layers)
            accs = [acc_by_layer[n] for n in layers]
            best_acc = max(accs)
            ax.plot(
                layers, accs,
                color=colors[label], marker=markers.get(label, "x"),
                lw=2.0, markersize=8, label=f"{label} ({best_acc * 100.0:.1f}%)",
            )

        if not all_layers:
            raise ValueError("plot_accuracy_vs_layers requires at least one data point")
        min_layer = min(all_layers)
        max_layer = max(all_layers)
        ax.set_title("Performance V.S. Layer Number", fontsize=16, fontweight="bold")
        ax.set_xlabel("Layer Number", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_xlim(min_layer, max_layer)
        ax.set_ylim(*ylim)
        ax.set_xticks(sorted(all_layers))
        yticks = [round(v, 1) for v in np.arange(ylim[0], ylim[1] + 0.001, 0.1)]
        ax.set_yticks(yticks)
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=12, width=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        legend = ax.legend(
            loc="lower right", fontsize=11, frameon=True, fancybox=False,
            framealpha=1.0, edgecolor="#1f1f1f",
        )
        legend._legend_box.align = "left"
        return self._save(fig, name)

    def plot_s7b_convergence_with_schematics(
        self,
        curves: dict[str, list[float]],
        max_acc: dict[str, float],
        *,
        ordered_labels: list[str],
        colors: dict[str, str],
        schematic_keys: list[str],
        schematic_num_layers: int | list[int] = 5,
        name: str = "s7b_sbn_position.png",
        ylim: tuple[float, float] = (0.8, 1.0),
    ) -> Path:
        """S7(b)-style: convergence curves with schematic insets."""
        from tao2019_fd2nn.viz.d2nn_schematic import draw_s7_schematic

        if isinstance(schematic_num_layers, int):
            schematic_layers = [int(schematic_num_layers)] * len(schematic_keys)
        else:
            schematic_layers = [int(v) for v in schematic_num_layers]
        if len(schematic_layers) != len(schematic_keys):
            raise ValueError("schematic_num_layers must match schematic_keys length")

        fig = plt.figure(figsize=(8.0, 9.5), dpi=150)
        fig.patch.set_facecolor("white")
        n_sch = len(schematic_keys)
        n_cols = (n_sch + 1) // 2  # 2 per row
        n_rows_sch = 2 if n_sch > 2 else 1
        gs = gridspec.GridSpec(
            n_rows_sch + 1, max(n_cols, 1),
            height_ratios=[1] * n_rows_sch + [2.5], hspace=0.20, wspace=0.08,
            left=0.10, right=0.95, top=0.96, bottom=0.06,
        )
        for i, sk in enumerate(schematic_keys):
            r = i // n_cols
            c = i % n_cols
            ax_s = fig.add_subplot(gs[r, c])
            draw_s7_schematic(ax_s, sk, num_layers=schematic_layers[i])

        ax = fig.add_subplot(gs[n_rows_sch, :])
        ax.set_facecolor("white")

        max_len = 0
        for label in ordered_labels:
            values = [float(v) for v in curves[label]]
            max_len = max(max_len, len(values))
            curve_max = float(max_acc.get(label, max(values)))
            y = np.array([0.0, *values], dtype=float)
            x = np.arange(0, len(values) + 1)
            ax.plot(x, y, color=colors[label], lw=2.0, label=f"{label} ({curve_max * 100.0:.1f}%)")

        x_max = max(30, max_len)
        ax.set_title("SBN Position", fontsize=16, fontweight="bold")
        ax.set_xlabel("Epoch Number", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_xlim(0, x_max)
        ax.set_ylim(*ylim)
        if x_max == 30:
            ax.set_xticks([0, 10, 20, 30])
        yticks = [round(v, 1) for v in np.arange(ylim[0], ylim[1] + 0.001, 0.1)]
        ax.set_yticks(yticks)
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=12, width=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        legend = ax.legend(
            loc="lower right", fontsize=10, frameon=True, fancybox=False,
            framealpha=1.0, edgecolor="#1f1f1f",
        )
        legend._legend_box.align = "left"
        return self._save(fig, name)

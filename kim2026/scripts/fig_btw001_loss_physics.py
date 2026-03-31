"""btw-001: Why irradiance loss works best — physics visualization.

Compares amplitude, phase, and intensity across all loss strategies
to show the Loss-Physics Alignment principle.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9,
    "figure.dpi": 180,
    "figure.facecolor": "white",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
})

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
FIG_DIR = RUNS_DIR / "figures_sweep_report"

STRATEGIES = {
    "Complex\n(co+amp)": "02_fd2nn_complexloss_roi1024_phase_range_sweep_claude/tanh_2pi",
    "Irradiance\n(io+br+ee)": "04_fd2nn_irradianceloss_roi1024_phase_range_sweep_claude/tanh_2pi",
}


def load_complex(npz_path: Path) -> dict:
    npz = np.load(npz_path)
    return {
        "input": npz["input_real"] + 1j * npz["input_imag"],
        "pred": npz["pred_real"] + 1j * npz["pred_imag"],
        "target": npz["target_real"] + 1j * npz["target_imag"],
    }


def center_crop(arr: np.ndarray, radius: int) -> np.ndarray:
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    return arr[cy - radius:cy + radius, cx - radius:cx + radius]


def align_global_phase(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    inner = np.sum(pred * np.conj(target))
    alpha = np.angle(inner)
    return pred * np.exp(-1j * alpha)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data = {}
    for label, rel_path in STRATEGIES.items():
        npz_path = RUNS_DIR / rel_path / "sample_fields.npz"
        if not npz_path.exists():
            print(f"SKIP {label}: {npz_path} not found")
            continue
        data[label] = load_complex(npz_path)

    if not data:
        print("No data found!")
        return

    target = next(iter(data.values()))["target"]
    turbulent_input = next(iter(data.values()))["input"]

    R = 200  # crop radius for zoom

    # ─────────────────────────────────────────────────
    # Figure 1: THE KEY EVIDENCE — amplitude vs intensity comparison
    # ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))

    cols = ["Turbulent\nInput", "Vacuum\nTarget", "Complex\n(co+amp)", "Irradiance\n(io+br+ee)"]
    fields = [turbulent_input, target]
    for label in STRATEGIES:
        if label in data:
            fields.append(data[label]["pred"])

    target_crop = center_crop(target, R)
    ref_max = float((np.abs(target_crop) ** 2).max())
    target_amp_max = float(np.abs(target_crop).max())

    # Row 0: Amplitude |E|
    for j, (col_label, field) in enumerate(zip(cols, fields)):
        fc = center_crop(field, R)
        amp = np.abs(fc) / target_amp_max
        im = axes[0, j].imshow(amp, origin="lower", cmap="magma", vmin=0, vmax=1.2)
        axes[0, j].set_title(col_label, fontsize=9, fontweight="bold")
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        # annotate RMSE vs target
        if j >= 2:
            amp_rmse = float(np.sqrt(np.mean((np.abs(fc) - np.abs(target_crop)) ** 2)) / target_amp_max)
            axes[0, j].text(0.02, 0.98, f"amp RMSE={amp_rmse:.3f}",
                            transform=axes[0, j].transAxes, fontsize=8,
                            va="top", ha="left", color="white",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
    axes[0, 0].set_ylabel("Amplitude |E|", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.7, pad=0.01, label="|E| / |E_target|_max")

    # Row 1: Intensity |E|²
    for j, (col_label, field) in enumerate(zip(cols, fields)):
        fc = center_crop(field, R)
        intensity = np.abs(fc) ** 2 / ref_max
        im = axes[1, j].imshow(intensity, origin="lower", cmap="inferno", vmin=0, vmax=1.0)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        if j >= 2:
            # intensity overlap
            pred_i = np.abs(fc) ** 2
            tgt_i = np.abs(target_crop) ** 2
            io_val = float(np.sum(pred_i * tgt_i) /
                           max(np.linalg.norm(pred_i.ravel()) * np.linalg.norm(tgt_i.ravel()), 1e-12))
            axes[1, j].text(0.02, 0.98, f"io={io_val:.3f}",
                            transform=axes[1, j].transAxes, fontsize=8,
                            va="top", ha="left", color="white",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
    axes[1, 0].set_ylabel("Intensity |E|²", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.7, pad=0.01, label="I / I_target_max")

    # Row 2: Phase (masked)
    for j, (col_label, field) in enumerate(zip(cols, fields)):
        fc = center_crop(field, R)
        phase = np.angle(fc)
        mask = np.abs(fc) ** 2 < ref_max * 0.01
        phase_ma = np.ma.masked_where(mask, phase)
        im = axes[2, j].imshow(phase_ma, origin="lower", cmap="twilight_shifted",
                                vmin=-math.pi, vmax=math.pi)
        axes[2, j].set_xticks([]); axes[2, j].set_yticks([])
        if j >= 2:
            aligned = align_global_phase(fc, target_crop)
            phase_diff = np.angle(aligned) - np.angle(target_crop)
            phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
            both_mask = (np.abs(fc) ** 2 > ref_max * 0.01) & (np.abs(target_crop) ** 2 > ref_max * 0.01)
            if both_mask.any():
                pr_val = float(np.sqrt(np.mean(phase_diff[both_mask] ** 2)))
            else:
                pr_val = float("nan")
            axes[2, j].text(0.02, 0.98, f"pr={pr_val:.3f} rad",
                            transform=axes[2, j].transAxes, fontsize=8,
                            va="top", ha="left", color="white",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
    axes[2, 0].set_ylabel("Phase ∠E", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axes[2, :].tolist(), shrink=0.7, pad=0.01, label="Phase [rad]")

    fig.suptitle("btw-001: Loss-Physics Alignment — Why Irradiance Loss Wins\n"
                 "amp RMSE ≈ same (phase-only can't control amplitude), "
                 "but intensity overlap differs 2.5×",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    path1 = FIG_DIR / "fig8_btw001_loss_physics_evidence.png"
    fig.savefig(path1, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path1}")

    # ─────────────────────────────────────────────────
    # Figure 2: Amplitude difference maps + centerline profiles
    # ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    target_crop = center_crop(target, R)
    target_amp = np.abs(target_crop) / target_amp_max
    target_i = np.abs(target_crop) ** 2 / ref_max

    strategy_labels = list(STRATEGIES.keys())
    for j, label in enumerate(strategy_labels):
        if label not in data:
            continue
        pred_crop = center_crop(data[label]["pred"], R)
        pred_amp = np.abs(pred_crop) / target_amp_max
        pred_i = np.abs(pred_crop) ** 2 / ref_max

        # Row 0: Amplitude difference |E_pred| - |E_target|
        amp_diff = pred_amp - target_amp
        im0 = axes[0, j].imshow(amp_diff, origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        amp_rmse = float(np.sqrt(np.mean(amp_diff ** 2)))
        axes[0, j].set_title(f"{label}\namp ΔRMSE = {amp_rmse:.4f}", fontsize=9)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])

        # Row 1: Intensity difference |E_pred|² - |E_target|²
        i_diff = pred_i - target_i
        im1 = axes[1, j].imshow(i_diff, origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        i_rmse = float(np.sqrt(np.mean(i_diff ** 2)))
        axes[1, j].set_title(f"int ΔRMSE = {i_rmse:.4f}", fontsize=9)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])

    # Profiles in the rightmost column
    axes[0, 2].set_visible(True)
    axes[1, 2].set_visible(True)
    cx = R  # center of cropped field
    x = np.arange(2 * R) - R

    # Amplitude centerline
    axes[0, 2].plot(x, np.abs(center_crop(target, R))[cx] / target_amp_max,
                     "k-", lw=2, label="Target")
    axes[0, 2].plot(x, np.abs(center_crop(turbulent_input, R))[cx] / target_amp_max,
                     "gray", lw=1.5, alpha=0.5, label="Turbulent")
    colors = ["#e74c3c", "#2ecc71"]
    for k, label in enumerate(strategy_labels):
        if label not in data:
            continue
        pred_crop = center_crop(data[label]["pred"], R)
        axes[0, 2].plot(x, np.abs(pred_crop)[cx] / target_amp_max,
                         color=colors[k], lw=1.5, label=label.replace("\n", " "))
    axes[0, 2].set_title("Amplitude Centerline", fontsize=9)
    axes[0, 2].set_xlabel("Pixel offset")
    axes[0, 2].set_ylabel("|E| / |E_target|_max")
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(-0.05, 1.3)

    # Intensity centerline
    axes[1, 2].plot(x, (np.abs(center_crop(target, R))[cx]) ** 2 / ref_max,
                     "k-", lw=2, label="Target")
    axes[1, 2].plot(x, (np.abs(center_crop(turbulent_input, R))[cx]) ** 2 / ref_max,
                     "gray", lw=1.5, alpha=0.5, label="Turbulent")
    for k, label in enumerate(strategy_labels):
        if label not in data:
            continue
        pred_crop = center_crop(data[label]["pred"], R)
        axes[1, 2].plot(x, (np.abs(pred_crop)[cx]) ** 2 / ref_max,
                         color=colors[k], lw=1.5, label=label.replace("\n", " "))
    axes[1, 2].set_title("Intensity Centerline", fontsize=9)
    axes[1, 2].set_xlabel("Pixel offset")
    axes[1, 2].set_ylabel("I / I_target_max")
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(-0.05, 1.3)

    axes[0, 0].set_ylabel("Amplitude Error\n|E_pred| − |E_target|", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Intensity Error\nI_pred − I_target", fontsize=10, fontweight="bold")

    fig.colorbar(im0, ax=axes[0, :2].tolist(), shrink=0.8, pad=0.02, label="Δ (red=over, blue=under)")
    fig.colorbar(im1, ax=axes[1, :2].tolist(), shrink=0.8, pad=0.02, label="Δ (red=over, blue=under)")

    fig.suptitle("btw-001: Amplitude Error ≈ Same, Intensity Error Vastly Different\n"
                 "Phase-only mask redistributes energy (intensity) without changing amplitude",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    path2 = FIG_DIR / "fig9_btw001_amplitude_vs_intensity_error.png"
    fig.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path2}")

    # ─────────────────────────────────────────────────
    # Figure 3: Phase error comparison
    # ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for j, label in enumerate(strategy_labels):
        if label not in data:
            continue
        pred_crop = center_crop(data[label]["pred"], R)
        aligned = align_global_phase(pred_crop, target_crop)

        phase_diff = np.angle(aligned) - np.angle(target_crop)
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

        both_above = (np.abs(pred_crop) ** 2 > ref_max * 0.01) & (np.abs(target_crop) ** 2 > ref_max * 0.01)
        phase_diff_masked = np.ma.masked_where(~both_above, phase_diff)

        # Row 0: Phase error 2D map
        im = axes[0, j].imshow(phase_diff_masked, origin="lower", cmap="twilight_shifted",
                                vmin=-math.pi, vmax=math.pi)
        pr_val = float(np.sqrt(np.mean(phase_diff[both_above] ** 2))) if both_above.any() else float("nan")
        axes[0, j].set_title(f"{label}\nPhase RMSE = {pr_val:.3f} rad", fontsize=9)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])

        # Row 1: Phase error histogram
        valid = phase_diff[both_above] if both_above.any() else np.array([0.0])
        axes[1, j].hist(valid.ravel(), bins=100, range=(-math.pi, math.pi),
                         density=True, alpha=0.7, color=colors[j], edgecolor="none")
        axes[1, j].axvline(0, color="black", lw=1, ls="--")
        axes[1, j].set_xlabel("Phase error [rad]")
        axes[1, j].set_ylabel("Density")
        axes[1, j].set_title(f"std = {float(np.std(valid)):.3f} rad", fontsize=9)
        axes[1, j].set_xlim(-math.pi, math.pi)
        axes[1, j].grid(True, alpha=0.3)

    # Phase error centerline in the right column
    axes[0, 2].set_visible(True)
    axes[1, 2].set_visible(True)

    for k, label in enumerate(strategy_labels):
        if label not in data:
            continue
        pred_crop = center_crop(data[label]["pred"], R)
        aligned = align_global_phase(pred_crop, target_crop)
        phase_diff = np.angle(aligned) - np.angle(target_crop)
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        centerline = phase_diff[R]
        axes[0, 2].plot(x, centerline, color=colors[k], lw=1.5,
                         label=label.replace("\n", " "))
    axes[0, 2].axhline(0, color="black", lw=1, ls="--")
    axes[0, 2].set_title("Phase Error Centerline", fontsize=9)
    axes[0, 2].set_xlabel("Pixel offset")
    axes[0, 2].set_ylabel("Δφ [rad]")
    axes[0, 2].set_ylim(-math.pi, math.pi)
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].grid(True, alpha=0.3)

    # Turbulent input phase error
    input_crop = center_crop(turbulent_input, R)
    input_aligned = align_global_phase(input_crop, target_crop)
    input_pdiff = np.angle(input_aligned) - np.angle(target_crop)
    input_pdiff = (input_pdiff + np.pi) % (2 * np.pi) - np.pi
    input_mask = (np.abs(input_crop) ** 2 > ref_max * 0.01) & (np.abs(target_crop) ** 2 > ref_max * 0.01)
    input_valid = input_pdiff[input_mask] if input_mask.any() else np.array([0.0])

    axes[1, 2].hist(input_valid.ravel(), bins=100, range=(-math.pi, math.pi),
                     density=True, alpha=0.4, color="gray", edgecolor="none", label="Turbulent")
    for k, label in enumerate(strategy_labels):
        if label not in data:
            continue
        pred_crop = center_crop(data[label]["pred"], R)
        aligned = align_global_phase(pred_crop, target_crop)
        phase_diff = np.angle(aligned) - np.angle(target_crop)
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        both_above = (np.abs(pred_crop) ** 2 > ref_max * 0.01) & (np.abs(target_crop) ** 2 > ref_max * 0.01)
        valid = phase_diff[both_above] if both_above.any() else np.array([0.0])
        axes[1, 2].hist(valid.ravel(), bins=100, range=(-math.pi, math.pi),
                         density=True, alpha=0.5, color=colors[k], edgecolor="none",
                         label=label.replace("\n", " "))
    axes[1, 2].axvline(0, color="black", lw=1, ls="--")
    axes[1, 2].set_title("Phase Error Distribution Comparison", fontsize=9)
    axes[1, 2].set_xlabel("Phase error [rad]")
    axes[1, 2].set_ylabel("Density")
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Phase Error Map\nΔφ = ∠E_pred − ∠E_target", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Phase Error\nHistogram", fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=axes[0, :2].tolist(), shrink=0.8, pad=0.02, label="Phase error [rad]")

    fig.suptitle("btw-001: Phase Correction — Complex Loss ≫ Irradiance Loss\n"
                 "Complex loss concentrates phase errors near 0; Irradiance loss ignores phase entirely",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    path3 = FIG_DIR / "fig10_btw001_phase_error_comparison.png"
    fig.savefig(path3, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path3}")

    print("\nDone! 3 figures generated.")


if __name__ == "__main__":
    main()

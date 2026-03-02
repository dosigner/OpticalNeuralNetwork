"""Visualization exports."""

from .classifier import (
    plot_confusion_matrix,
    plot_energy_distribution_heatmap,
    plot_inference_summary,
    plot_output_with_detectors,
)
from .fields import plot_intensity, plot_phase_mask
from .imaging import compute_ssim, plot_imaging_comparison
from .propagation import (
    extract_xz_cross_section,
    generate_phase_masks,
    make_fresnel_lens_phase,
    plot_propagation_stack,
    plot_stacked_xy_comparison,
    plot_wave_propagation_figure_s6,
    plot_xz_cross_section,
    plot_xz_cross_section_comparison,
    plot_xz_cross_section_volume,
    simulate_d2nn_volume,
    simulate_free_space_volume,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_energy_distribution_heatmap",
    "plot_inference_summary",
    "plot_output_with_detectors",
    "plot_intensity",
    "plot_phase_mask",
    "compute_ssim",
    "plot_imaging_comparison",
    "make_fresnel_lens_phase",
    "generate_phase_masks",
    "simulate_free_space_volume",
    "simulate_d2nn_volume",
    "extract_xz_cross_section",
    "plot_xz_cross_section_volume",
    "plot_xz_cross_section_comparison",
    "plot_stacked_xy_comparison",
    "plot_wave_propagation_figure_s6",
    "plot_propagation_stack",
    "plot_xz_cross_section",
]

from __future__ import annotations


FIGURE_TEXTS = {
    "fig2": {
        "suptitle": "Figure 2: Experimental Verification of Unitary Invariance\nCO and WF RMS remain stable during training, while PIB changes",
        "titles": [
            "(a) CO — preserved (Theorem 1)",
            "(b) WF RMS — preserved (Theorem 2)",
            "(c) PIB — changes (nonlinear metric)",
        ],
        "wf_annotation": "460 ± 5 nm\n(invariant across all strategies)",
    },
    "fig3": {
        "suptitle": "Figure 3: Deterministic Aberration Correction vs Random Turbulence",
        "left_title": "(a) Deterministic aberration (Defocus Z4)\nSingle-layer D2NN — perfect correction",
        "left_legend": ["WF RMS (212→0.1 nm)", "CO (0.989→1.000)"],
        "right_wf_label": "WF RMS (460 nm, invariant)",
        "right_title": "(b) Random turbulence (D/r₀=5.02)\n5-layer D2NN — no correction",
        "right_annotation": "WF RMS invariant\n(Theorem 2)",
    },
    "fig4": {
        "suptitle": "Figure 4: Performance Across Loss Strategies\nPIB can improve while CO and WF RMS remain constrained",
        "titles": [
            "(a) Focusing efficiency (PIB)",
            "(b) Complex overlap (CO)",
            "(c) Wavefront error (WF RMS)",
        ],
    },
    "fig5": {
        "suptitle": "Figure 5: CO–PIB Trade-off (Pareto frontier)\nImproving the nonlinear metric (PIB) can destroy the linear metric (CO)",
        "xlabel": "Complex Overlap (CO)",
        "ylabel": "PIB@50μm",
        "ideal_region": "Ideal region\n(CO up, PIB up)",
        "filtering_region": "PIB up, CO down\n(spatial filtering)",
    },
    "fig6": {
        "suptitle": "Figure 6: Spatial Energy Redistribution at the Detector Plane\nThe D2NN concentrates intensity into the 50μm bucket",
        "field_labels": [
            "Vacuum (target)",
            "Turbulent (uncorrected)",
            "PIB only",
            "CO+PIB hybrid",
        ],
        "row_labels": [
            "Irradiance\n(linear)",
            "Irradiance\n(log scale)",
            "Difference map\n(vs vacuum)",
        ],
    },
}

"""Generate application roadmap figure for Section 7."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def make_fig_application_roadmap(save_path: str = "docs/images/fig_application_roadmap.png") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), facecolor="white")

    # TRL levels on x-axis, application domains on y-axis
    trl_labels = [
        "TRL 1-2\nBasic\nResearch",
        "TRL 3-4\nLab\nValidation",
        "TRL 5-6\nPrototype\nSystem",
        "TRL 7-8\nPre-production",
        "TRL 9\nDeployment",
    ]
    trl_x = np.arange(len(trl_labels))

    domains = [
        "Security\nImaging",
        "Medical\nImaging",
        "THz\nCommunication",
        "Industrial\nInspection",
        "Scientific\nInstrumentation",
    ]
    domain_y = np.arange(len(domains))

    # Current status and target for each domain
    # (domain_idx, current_trl_idx, target_trl_idx, color, difficulty)
    roadmap_data = [
        # Security: simulation done (TRL2), lab validation next, product feasible
        (0, 1, 4, "#2196F3", "Medium"),
        # Medical: simulation done, needs tissue phantoms
        (1, 0, 3, "#4CAF50", "High"),
        # THz Comm: early research
        (2, 0, 2, "#FF9800", "High"),
        # Industrial: closest to deployment in THz regime
        (3, 1, 4, "#9C27B0", "Medium"),
        # Scientific: already useful as research tool
        (4, 1, 3, "#F44336", "Low"),
    ]

    # Draw grid
    for x in trl_x:
        ax.axvline(x, color="#e0e0e0", linewidth=0.8, zorder=0)
    for y in domain_y:
        ax.axhline(y, color="#e0e0e0", linewidth=0.8, zorder=0)

    # Draw roadmap arrows and markers
    arrow_height = 0.12
    for domain_idx, current_trl, target_trl, color, difficulty in roadmap_data:
        y = domain_idx

        # Current position marker
        ax.scatter(current_trl, y, s=200, color=color, zorder=5, edgecolors="black", linewidths=1.5)
        ax.annotate(
            "Current",
            (current_trl, y),
            textcoords="offset points",
            xytext=(0, -22),
            ha="center",
            fontsize=7,
            color=color,
            fontweight="bold",
        )

        # Arrow from current to target
        if target_trl > current_trl:
            ax.annotate(
                "",
                xy=(target_trl - 0.08, y),
                xytext=(current_trl + 0.15, y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=2.5,
                    linestyle="--" if difficulty == "High" else "-",
                    mutation_scale=18,
                ),
                zorder=4,
            )

        # Target marker (star)
        ax.scatter(
            target_trl, y, s=300, color=color, marker="*",
            zorder=5, edgecolors="black", linewidths=0.8,
        )

        # Difficulty label
        diff_colors = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}
        ax.text(
            target_trl + 0.15, y + 0.18,
            difficulty,
            fontsize=7,
            color=diff_colors[difficulty],
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    # Key milestones as text boxes
    milestones = [
        (1, -0.85, "This work:\nSimulation\nreproduction\ncomplete", "#1565C0"),
        (2, -0.85, "Next:\n3D-print\nphase plates\n+ bench test", "#2E7D32"),
        (3, -0.85, "Future:\nReal-time\ndemonstration\nwith THz source", "#E65100"),
    ]
    for mx, my, text, color in milestones:
        ax.text(
            mx, my, text,
            fontsize=7.5,
            color=color,
            ha="center",
            va="top",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=color,
                alpha=0.1,
                edgecolor=color,
                linewidth=1.5,
            ),
        )

    # Axes configuration
    ax.set_xticks(trl_x)
    ax.set_xticklabels(trl_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(domain_y)
    ax.set_yticklabels(domains, fontsize=11, fontweight="bold")

    ax.set_xlim(-0.5, len(trl_labels) - 0.5)
    ax.set_ylim(-1.5, len(domains) - 0.5)

    ax.set_xlabel("Technology Readiness Level (TRL)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        "Application Roadmap: Random-Diffuser D2NN\nFrom Simulation to Deployment",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                    markersize=10, markeredgecolor="black", label="Current status"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                    markersize=14, markeredgecolor="black", label="Target status"),
        plt.Line2D([0], [0], color="gray", linewidth=2.5, linestyle="-", label="Feasible path"),
        plt.Line2D([0], [0], color="gray", linewidth=2.5, linestyle="--", label="Challenging path"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#666666")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    make_fig_application_roadmap()

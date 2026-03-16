"""Generate tacit knowledge map infographic for Section 6."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(18, 13), facecolor="white")
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis("off")

# Title
ax.text(9.0, 12.5, "Tacit Knowledge Map: Luo et al. 2022 D2NN Reproduction",
        fontsize=18, fontweight="bold", ha="center", va="center",
        fontfamily="sans-serif")
ax.text(9.0, 12.1, "30 undocumented implementation choices discovered during reproduction",
        fontsize=11, ha="center", va="center", color="#555555", style="italic")

# Category definitions
categories = [
    {
        "title": "6.1 Physics Modeling\n(Hidden Assumptions)",
        "color": "#E53935",
        "items": ["TK-1: Scalar approx.", "TK-2: Coherent illum.",
                  "TK-3: Evanescent mask", "TK-4: Asymmetric z"],
        "x": 1.5, "y": 9.5, "w": 4.5, "h": 2.2,
    },
    {
        "title": "6.2 Numerical Impl.\n(Non-trivial Choices)",
        "color": "#1E88E5",
        "items": ["TK-5: Zero-pad 2x", "TK-6: fftfreq order",
                  "TK-7: f64 precision", "TK-8: Reflect pad",
                  "TK-9: TF caching"],
        "x": 6.5, "y": 9.5, "w": 4.5, "h": 2.2,
    },
    {
        "title": "6.3 Training Pipeline\n(Implicit Decisions)",
        "color": "#43A047",
        "items": ["TK-10: B*n batch", "TK-11: Epoch refresh",
                  "TK-12: Seed scheme", "TK-13: Loss structure",
                  "TK-14: Global PCC", "TK-15: LR gamma=0.99"],
        "x": 12.0, "y": 9.5, "w": 4.5, "h": 2.2,
    },
    {
        "title": "6.4 Diffuser Physics\n(Hidden Generation)",
        "color": "#FB8C00",
        "items": ["TK-16: Height stats", "TK-17: Sigma-L relation",
                  "TK-18: Corr. fitting"],
        "x": 1.5, "y": 5.8, "w": 4.5, "h": 2.2,
    },
    {
        "title": "6.5 Sensitivity Analysis\n(Discovered in Repro.)",
        "color": "#8E24AA",
        "items": ["TK-19: n-sweep saturation", "TK-20: Known<New reversal",
                  "TK-21: Period instability", "TK-22: Contrast vs PCC",
                  "TK-23: Amplitude enc."],
        "x": 6.5, "y": 5.8, "w": 4.5, "h": 2.2,
    },
    {
        "title": "6.6 Paper Ambiguities\n(vs Actual Impl.)",
        "color": "#00897B",
        "items": ["TK-24: Dual n meaning", "TK-25: OOD targets",
                  "TK-26: Pruning masks", "TK-27: Phase wrap",
                  "TK-28: 3-stage resize", "TK-29: Blind seeds",
                  "TK-30: Corr.len. change"],
        "x": 12.0, "y": 5.8, "w": 4.5, "h": 2.2,
    },
]

for cat in categories:
    # Box
    rect = mpatches.FancyBboxPatch(
        (cat["x"], cat["y"] - cat["h"]),
        cat["w"], cat["h"],
        boxstyle="round,pad=0.15",
        facecolor=cat["color"] + "18",
        edgecolor=cat["color"],
        linewidth=2.5,
    )
    ax.add_patch(rect)

    # Title
    ax.text(cat["x"] + cat["w"] / 2, cat["y"] - 0.15,
            cat["title"], fontsize=10, fontweight="bold",
            ha="center", va="top", color=cat["color"],
            fontfamily="sans-serif")

    # Items
    for i, item in enumerate(cat["items"]):
        y_pos = cat["y"] - 0.75 - i * 0.22
        ax.text(cat["x"] + 0.3, y_pos, item,
                fontsize=7.5, ha="left", va="center",
                color="#333333", fontfamily="monospace")

# Impact summary at bottom
impact_box = mpatches.FancyBboxPatch(
    (1.5, 0.5), 15.0, 2.5,
    boxstyle="round,pad=0.2",
    facecolor="#F5F5F5",
    edgecolor="#999999",
    linewidth=1.5,
)
ax.add_patch(impact_box)

ax.text(9.0, 2.75, "Impact on Reproduction Outputs",
        fontsize=13, fontweight="bold", ha="center", va="center",
        color="#333333")

outputs = ["Fig.2 (Known/New)", "Fig.3 (Period Sweep)",
           "FigS4 (Pruning)", "FigS5 (Corr.Length)", "Training Loop"]
counts = [12, 11, 10, 13, 18]
colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA"]

bar_y = 1.4
bar_h = 0.5
total_w = 13.0
start_x = 2.5

for i, (label, count, color) in enumerate(zip(outputs, counts, colors)):
    w = total_w / len(outputs)
    x = start_x + i * w
    # Bar
    bar_w = (count / 30) * (w * 0.7)
    rect = mpatches.FancyBboxPatch(
        (x + 0.1, bar_y - bar_h / 2),
        bar_w, bar_h,
        boxstyle="round,pad=0.05",
        facecolor=color + "CC",
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(x + 0.1 + bar_w + 0.15, bar_y,
            f"{count}/30", fontsize=9, fontweight="bold",
            ha="left", va="center", color=color)
    ax.text(x + w / 2, bar_y - 0.55, label,
            fontsize=8, ha="center", va="center", color="#555555")

# Connecting lines from categories to impact bar
for cat in categories:
    cx = cat["x"] + cat["w"] / 2
    cy = cat["y"] - cat["h"]
    ax.annotate("", xy=(cx, 3.2), xytext=(cx, cy - 0.1),
                arrowprops=dict(arrowstyle="->", color=cat["color"] + "66",
                               lw=1.5, connectionstyle="arc3,rad=0"))

fig.tight_layout(pad=0.5)
fig.savefig("/root/dj/D2NN/docs/images/fig_tacit_knowledge_map.png",
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: /root/dj/D2NN/docs/images/fig_tacit_knowledge_map.png")

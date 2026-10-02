"""Static L12 figure: the three spring-chain scenarios.

Run from the repository root: uv run --locked python scripts/l12_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"
FIXED, RED, BLUE, SPRING = "#8c8c8c", "#e06666", "#6fb7d9", "#a6a6a6"
DPI = 300
R = 0.32


def spring(ax, x0, x1, y, lw=2.0, color=SPRING, coils=8, amp=0.16, ls="-"):
    lead = 0.12 * (x1 - x0)
    xs = np.linspace(x0 + lead, x1 - lead, 2 * coils + 1)
    ys = y + amp * np.array([0] + [(-1) ** i for i in range(1, 2 * coils)] + [0])
    ax.plot([x0, xs[0]], [y, y], color=color, lw=lw, ls=ls, zorder=1)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=1)
    ax.plot([xs[-1], x1], [y, y], color=color, lw=lw, ls=ls, zorder=1)


def atom(ax, x, y, color, label):
    ax.add_patch(Circle((x, y), R, color=color, zorder=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=13, zorder=3)


def three_scenarios():
    fig, axes = plt.subplots(
        1, 3, figsize=(12.5, 2.2), layout="constrained",
        gridspec_kw={"width_ratios": [4.8, 4.8, 6.35]},
    )
    y = 0.0

    ax = axes[0]
    ax.plot([0, 2], [y, y], color="#c8c8c8", lw=1.5, ls=(0, (3, 3)), zorder=1)
    spring(ax, 2, 4, y)
    atom(ax, 0, y, FIXED, "0")
    atom(ax, 2, y, RED, "1")
    atom(ax, 4, y, BLUE, "2")
    ax.text(1, 0.45, r"$k_{01}=0$ (removed)", ha="center", fontsize=11)
    ax.text(3, 0.45, r"$k_{12}$", ha="center", fontsize=11)
    ax.set_title("1. Missing spring: singular", fontsize=12)

    ax = axes[1]
    spring(ax, 0, 2, y, lw=0.7)
    spring(ax, 2, 4, y)
    atom(ax, 0, y, FIXED, "0")
    atom(ax, 2, y, RED, "1")
    atom(ax, 4, y, BLUE, "2")
    ax.text(1, 0.45, r"$k_{01}\ll k_{12}$", ha="center", fontsize=11)
    ax.text(3, 0.45, r"$k_{12}$", ha="center", fontsize=11)
    ax.set_title("2. Weak spring: ill-conditioned", fontsize=12)

    ax = axes[2]
    xs = [0, 1.15, 2.3, 3.45]
    for a, b in zip(xs[:-1], xs[1:]):
        spring(ax, a, b, y, coils=4, amp=0.13)
    ax.plot([3.45, 3.7], [y, y], color=SPRING, lw=2.0, zorder=1)
    ax.text(3.95, y, r"$\cdots$", ha="center", va="center", fontsize=16)
    spring(ax, 4.4, 5.55, y, coils=4, amp=0.13)
    for x, c, lab in zip(xs, [FIXED, RED, RED, RED], ["0", "1", "2", "3"]):
        atom(ax, x, y, c, lab)
    atom(ax, 5.55, y, BLUE, r"$N$")
    ax.text(2.8, 0.45, r"$N\sim10^6$ moving atoms", ha="center", fontsize=11)
    ax.set_title("3. Long chain: computational cost", fontsize=12)

    for ax in axes:
        ax.text(0, -0.5, "(fixed)", ha="center", va="top", fontsize=10)
        ax.set_aspect("equal")
        ax.set_ylim(-0.85, 0.8)
        ax.axis("off")
    axes[2].set_xlim(-0.4, 5.95)
    for ax in axes[:2]:
        ax.set_xlim(-0.4, 4.4)
    fig.savefig(ASSETS / "L12-three-scenarios.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    three_scenarios()


if __name__ == "__main__":
    main()

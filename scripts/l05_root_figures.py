"""Static L05 figures. Run from the repository root with uv run."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"
R, T, P = 0.08314462618, 280.0, 50.0
A, B = 3.592, 0.04267  # One mole; L² bar/mol² and L/mol.
GREEN, ORANGE = "#007c41", "#d87700"


def residual(volume):
    return R * T / (volume - B) - A / volume**2 - P


def main():
    plt.rcParams.update({"font.size": 12})
    volumes = np.linspace(0.15, 0.9, 500)
    ideal_volume = R * T / P
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(volumes, R * T / volumes, color=GREEN, lw=2.5, label=r"$P(V)=nRT/V$")
    ax.axhline(P, color="#444444", ls="--", label="Target: 50 bar")
    ax.plot([ideal_volume, ideal_volume], [0, P], color=ORANGE, ls=":")
    ax.scatter([ideal_volume], [P], color=ORANGE, s=65, zorder=4)
    ax.annotate(f"V = nRT/P = {ideal_volume:.3f} L", (ideal_volume, P),
                xytext=(0.49, 90), arrowprops={"arrowstyle": "->", "color": ORANGE})
    ax.set(xlabel="Volume (L)", ylabel="Pressure (bar)", ylim=(0, 170))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(ASSETS / "L05-ideal-gas.png", dpi=180)
    plt.close(fig)

    left, right = 0.2, 0.6
    fa, fb = residual(left), residual(right)
    midpoint = (left + right) / 2
    intercept = (left * fb - right * fa) / (fb - fa)
    volumes = np.linspace(left, right, 500)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True, layout="constrained")
    for ax, trial, title in zip(axes, [midpoint, intercept], ["Bisection", "False position"]):
        ax.plot(volumes, residual(volumes), color=GREEN, lw=2.5, label="F(V)")
        ax.axhline(0, color="#444444", lw=1)
        ax.scatter([left, right], [fa, fb], color="#444444", zorder=3)
        ax.axvline(trial, color=ORANGE, ls=":")
        ax.scatter([trial], [residual(trial)], color=ORANGE, s=50, zorder=4)
        if title == "False position":
            ax.plot([left, right], [fa, fb], color=ORANGE, ls="--", label="Line through endpoints")
        ax.set(title=title, xlabel="Volume (L)", ylim=(-21, 14))
        ax.text(trial + 0.01, -17, f"Trial:\n{trial:.3f} L", color=ORANGE)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Residual (bar)")
    axes[1].legend(fontsize=9)
    fig.savefig(ASSETS / "L05-false-position.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

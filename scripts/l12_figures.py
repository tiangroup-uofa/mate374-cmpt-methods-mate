"""Static L12 figure using the browser benchmark's power-law extrapolations.

Run from the repository root: uv run --locked python scripts/l12_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DPI = 300
COLOURS = {"inverse": "#931212", "solve": "#15428e"}
LABELS = {"inverse": "inv(K) @ f", "solve": "solve(K, f)"}
TARGET = 1_000_000
# Browser-run fit summary: exponent and estimated seconds at TARGET.
BROWSER_FITS = {
    "inverse": {"p": 3.083, "seconds": 1.45e9},
    "solve": {"p": 2.605, "seconds": 4.08e5},
}


def dense_benchmark():
    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    grid = np.linspace(0, TARGET, 500)
    for method, result in BROWSER_FITS.items():
        p = result["p"]
        t_target = result["seconds"]
        times = t_target * (grid / TARGET) ** p
        ax.plot(grid, times, color=COLOURS[method], lw=2,
                label=f"{LABELS[method]}: $p = {p:.3f}$")
        ax.plot(TARGET, t_target, "o", color=COLOURS[method], ms=6)
        duration = (f"{t_target / (365 * 86400):.1f} years" if method == "inverse"
                    else f"{t_target / 86400:.2f} days")
        ax.annotate(f"{t_target:.3g} s ({duration})", (TARGET, t_target),
                    xytext=(-8, -30 if method == "inverse" else 24),
                    textcoords="offset points", ha="right", color=COLOURS[method], fontsize=9)
    ax.axvline(TARGET, color="#9aa3ab", lw=0.8, ls=":")
    ax.set(xlim=(0, TARGET * 1.04), xlabel="Moving atoms $N$",
           ylabel="Extrapolated median time (s)",
           title="Dense spring chain: browser power-law fits")
    ax.title.set_fontsize(9)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(ROOT / "assets" / "L12-dense-benchmark.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def main():
    dense_benchmark()


if __name__ == "__main__":
    main()

"""Static L12 figure: dense inverse and solve timings with a power-law fit.

Run from the repository root: uv run --locked python scripts/l12_figures.py
Reads data/L12-dense-benchmark.json, written by scripts/l12_dense_benchmark.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DPI = 300
COLOURS = {"inverse": "#931212", "solve": "#15428e"}
LABELS = {"inverse": "inv(K) @ f", "solve": "solve(K, f)"}


def dense_benchmark():
    data = json.loads((ROOT / "data" / "L12-dense-benchmark.json").read_text())
    sizes = np.array(data["sizes"], dtype=float)
    target = 1e6
    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    grid = np.geomspace(sizes.min(), target, 200)
    for method, times in data["median_seconds"].items():
        p, log_c = np.polyfit(np.log(sizes), np.log(times), 1)
        t_target = np.exp(log_c + p * np.log(target))
        ax.loglog(sizes, times, "o", color=COLOURS[method], ms=6,
                  label=f"{LABELS[method]}: measured")
        ax.loglog(grid, np.exp(log_c + p * np.log(grid)), "--", color=COLOURS[method], lw=1.2,
                  label=f"fit $t = cN^{{p}}$, $p = {p:.2f}$")
        ax.plot(target, t_target, "s", color=COLOURS[method], mfc="white", ms=6)
        ax.annotate(f"{t_target / 86400:.1f} days", (target, t_target), xytext=(-8, 6),
                    textcoords="offset points", ha="right", color=COLOURS[method], fontsize=9)
    ax.axvline(target, color="#9aa3ab", lw=0.8, ls=":")
    ax.set(xlabel="Moving atoms $N$", ylabel="Median time (s)",
           title=f"Dense spring chain ({data['machine']})")
    ax.title.set_fontsize(9)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(ROOT / "assets" / "L12-dense-benchmark.png", dpi=DPI, facecolor="white")
    plt.close(fig)


def main():
    dense_benchmark()


if __name__ == "__main__":
    main()

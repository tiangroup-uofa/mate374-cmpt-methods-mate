"""Redraw the L04 speedup figure from saved timings; no benchmark is run."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_class_scaling(data):
    from matplotlib.ticker import FuncFormatter

    colors = ["#555555", "#c17c00", "#238443", "#0072b2", "#b5488c"]
    names = list(dict.fromkeys(r["method"] for r in data["records"]))
    target = data["target_N"]
    python_rows = {r["N"]: r["seconds"] for r in data["records"] if r["method"] == "Python"}
    python_max = max(python_rows)
    python_per_throw = python_rows[python_max] / python_max
    fig, (ax, summary) = plt.subplots(1, 2, figsize=(13, 6.5),
                                      gridspec_kw={"width_ratios": [3.5, 1.7]})
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.84, wspace=0.25)
    endpoints = []
    for label, color in zip(names, colors):
        rows = sorted([r for r in data["records"] if r["method"] == label], key=lambda r: r["N"])
        ns = np.array([r["N"] for r in rows])
        seconds = np.array([r["seconds"] for r in rows])
        reference = np.array([python_rows.get(int(n), python_per_throw * n) for n in ns])
        speedups = reference / seconds
        ax.loglog(ns, speedups, "o-", color=color, linewidth=2, markersize=5, label=label)
        if label != "Python":
            slowest = np.array([max(r["times"]) for r in rows])
            fastest = np.array([min(r["times"]) for r in rows])
            low, high = reference / slowest, reference / fastest
            ax.errorbar(ns, speedups, yerr=np.maximum(0, np.vstack([speedups - low, high - speedups])),
                        fmt="none", color=color, capsize=4, elinewidth=1.3, alpha=0.85)
        endpoint = float(python_per_throw / (seconds[-1] / ns[-1]))
        measured = int(ns[-1]) == target
        if not measured:
            ax.loglog([ns[-1], target], [speedups[-1], endpoint], "--", color=color, linewidth=1.8)
            ax.plot(target, endpoint, "o", markerfacecolor="white", markeredgecolor=color, markersize=6)
        endpoints.append((label, endpoint, measured, color))
    ax.set_xlabel("Number of throws, N", fontsize=12)
    ax.set_ylabel("Speedup relative to Python (×)", fontsize=12)
    ax.set_xlim(6e3, 2e12)
    ax.set_ylim(0.7, 1e5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:,.0f}×"))
    ax.grid(True, which="major", alpha=0.22)
    # The colored method labels in the summary serve as the legend.
    ax.set_title("Solid: method timed   |   Dashed: method extrapolated", fontsize=11)
    summary.axis("off")
    summary.set_title("At one trillion throws", loc="left", fontsize=13, fontweight="bold")
    for i, (label, speedup, measured, color) in enumerate(endpoints):
        y = 0.91 - i * 0.18
        summary.text(0, y, label, color=color, fontsize=11, fontweight="bold", transform=summary.transAxes)
        summary.text(0, y - 0.063, f"{speedup:,.0f}×" if speedup >= 100 else f"{speedup:.1f}×",
                     fontsize=16, fontweight="bold", transform=summary.transAxes)
        if label != "Python":
            status = "measured GPU runtime" if measured else "extrapolated method runtime"
            summary.text(0, y - 0.11, status, fontsize=9, transform=summary.transAxes)
    fig.suptitle("Buffon's needle: speedup from Python to GPU", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.08, 0.90, f'Molab: 4 CPUs  |  {data["gpu"]}', fontsize=10)
    fig.text(0.08, 0.10, "Speedup = Python time / method time. Python uses math.sin; its baseline is extrapolated beyond 10⁶ throws.", fontsize=9)
    fig.text(0.08, 0.065, "Median of 3 runs at every point. Whiskers: min–max method timing range; Python-reference uncertainty excluded.", fontsize=9)
    fig.text(0.08, 0.03, "GPU: shuffled repeat order, 1 s sustained warmup before each run. Batches: NumPy 10⁶, GPU 10⁷. Compilation excluded.", fontsize=9)
    return fig

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "data/L04-final-scaling.json").read_text())
    fig = plot_class_scaling(data)
    for extension in ("png", "svg", "pdf"):
        path = root / f"assets/L04-final-scaling.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(path)
    plt.close(fig)

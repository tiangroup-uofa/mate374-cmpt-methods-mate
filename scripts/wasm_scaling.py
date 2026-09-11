"""Generate WASM timing log-log plot for Lecture 04."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "assets" / "L04-wasm-scaling.png"

def main():
    N = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    t = np.array([0.0020, 0.0170, 0.0910, 0.6530, 7.1560, 78.6260])

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    ax.loglog(N, t, "o-", color="#e36209", markersize=7, linewidth=1.8,
              label="Measured (WASM)")

    N_ref = np.array([1e3, 1e9])
    t_ref = (0.653 / 1e6) * N_ref
    ax.loglog(N_ref, t_ref, "--", color="#888888", linewidth=1.0,
              label=r"Slope 1 ($t \propto N$)")

    N_extrap = 1e9
    t_extrap = 78.626 * 11
    ax.loglog(N_extrap, t_extrap, "s", color="#e36209", markersize=9,
              markerfacecolor="none", markeredgewidth=2,
              label=f"Extrapolated (~{t_extrap:.0f} s)")

    ax.set_xlabel("Number of throws $N$", fontsize=12)
    ax.set_ylabel("Elapsed time (s)", fontsize=12)
    ax.set_title("Buffon loop timing (WASM)", fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(5e2, 3e9)
    ax.set_ylim(5e-4, 3e3)

    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUTPUT}")

if __name__ == "__main__":
    main()

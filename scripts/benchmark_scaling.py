"""Generate benchmark comparison figure for Lecture 04."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "assets" / "L04-benchmark-scaling.png"

def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: execution time vs N (scaling)
    N = np.array([1e4, 1e5, 1e6, 1e7, 1e8])
    t_py = np.array([0.0035, 0.035, 0.35, 3.5, 35.0])
    t_np = np.array([0.0003, 0.0028, 0.027, 0.28, 2.9]) # batched
    t_numba = np.array([0.00008, 0.00075, 0.0074, 0.075, 0.74]) # warmed
    t_parallel = np.array([0.00025, 0.00045, 0.0018, 0.015, 0.14])
    t_gpu = np.array([0.004, 0.0042, 0.0055, 0.012, 0.045]) # launch overhead flat at small N

    ax1.loglog(N, t_py, 'o--', color='#d95f02', label='Pure Python loop', lw=1.5)
    ax1.loglog(N, t_np, 's-', color='#1b9e77', label='NumPy (batched)', lw=1.5)
    ax1.loglog(N, t_numba, '^-', color='#7570b3', label='Numba JIT (1 core)', lw=1.5)
    ax1.loglog(N, t_parallel, 'd-', color='#e7298a', label='Numba parallel (8 cores)', lw=1.5)
    ax1.loglog(N, t_gpu, '*-', color='#386cb0', label='PyTorch GPU (CUDA/MPS)', lw=2)

    ax1.set_xlabel("Number of throws $N$", fontsize=10)
    ax1.set_ylabel("Execution time (seconds)", fontsize=10)
    ax1.set_title("Runtime scaling across implementations", fontsize=11, fontweight='bold')
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8, frameon=True)

    # Right: Speedup relative to pure Python at N=10^7
    labels = ['Pure\nPython', 'NumPy\nvectorized', 'Numba\nJIT', 'Numba\nparallel', 'PyTorch\nGPU']
    speedups = [1.0, 3.5/0.28, 3.5/0.075, 3.5/0.015, 3.5/0.012]
    colors = ['#d95f02', '#1b9e77', '#7570b3', '#e7298a', '#386cb0']

    bars = ax2.bar(labels, speedups, color=colors, width=0.55, edgecolor='black', linewidth=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval * 1.15, f"{yval:.1f}×", 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_yscale('log')
    ax2.set_ylabel("Speedup relative to pure Python ($N=10^7$)", fontsize=10)
    ax2.set_title("Effective speedup at $10^7$ steps", fontsize=11, fontweight='bold')
    ax2.set_ylim(0.5, 1000)
    ax2.grid(True, axis='y', which="both", alpha=0.25)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200)
    print(f"Saved {OUTPUT}")

if __name__ == "__main__":
    main()

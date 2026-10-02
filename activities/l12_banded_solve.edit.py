# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import solve_banded
    from time import perf_counter
    return mo, np, perf_counter, plt, solve_banded


@app.cell
def _(mo):
    mo.md(r"""
    ## One million atoms with three stored diagonals
    The chain has one fixed end and an end load. A dense float64 matrix
    would need $8N^2$ bytes, while the three stored diagonals need only
    $24N$ bytes. The reference solution is $u_i=iF/k$.
    """)
    return


@app.cell
def _():
    N = 1_000_000
    k = 5.0
    F = 1e-7  # small end load keeps extensions small
    return F, N, k


@app.cell
def _(F, N, k, np, perf_counter, solve_banded):
    # Row 0 holds the upper diagonal, row 1 the main diagonal, row 2 the lower.
    K_band = np.zeros((3, N))
    K_band[0, 1:] = -k
    K_band[1, :] = 2 * k
    K_band[1, -1] = k
    K_band[2, :-1] = -k
    f = np.zeros(N)
    f[-1] = F
    start = perf_counter()
    u = solve_banded((1, 1), K_band, f)
    elapsed = perf_counter() - start
    print(f"Solve time (excluding assembly): {elapsed:.4g} s")
    print(f"Dense matrix alone: {8 * N**2 / 1e9:.4g} GB")
    print(f"Banded array alone: {K_band.nbytes / 1e6:.4g} MB")
    print(f"End displacement: {u[-1]:.8g}")
    return (u,)


@app.cell
def _(F, N, k, np):
    reference = np.arange(1, N + 1) * F / k
    return (reference,)


@app.cell
def _(N, np, plt, reference, u):
    _indices = np.unique(np.linspace(0, N - 1, min(N, 400), dtype=int))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(_indices + 1, reference[_indices], label="i F / k")
    ax.plot(_indices + 1, u[_indices], "--", label="Banded solve")
    ax.set(xlabel="Moving atom i", ylabel="Displacement")
    ax.legend()
    fig.tight_layout()
    fig
    return



if __name__ == "__main__":
    app.run()

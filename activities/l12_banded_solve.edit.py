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
    ## Can we store just the three nonzero diagonals?
    The chain has one fixed end and an end load. **Predict:** compare
    $8N^2$ bytes for a dense float64 matrix with $24N$ bytes for three
    stored diagonals. Edit `N` from 10,000 to 1,000,000 to measure a
    larger solve on your device. The reference solution is $u_i=iF/k$.
    """)
    return


@app.cell
def _():
    N = 10_000
    k = 5.0  # eV/Å²
    F = 1e-7  # eV/Å; small end load keeps extensions small
    return F, N, k


@app.cell
def _(F, N, k, np, perf_counter, solve_banded):
    # ab[0, 1:] is the upper diagonal; ab[2, :-1] is the lower.
    ab = np.zeros((3, N))
    ab[0, 1:] = -k
    ab[1, :] = 2 * k
    ab[1, -1] = k
    ab[2, :-1] = -k
    f = np.zeros(N)
    f[-1] = F
    start = perf_counter()
    u = solve_banded((1, 1), ab, f)
    elapsed = perf_counter() - start
    print(f"Solve time (excluding assembly): {elapsed:.4g} s")
    print(f"Dense matrix alone: {8 * N**2 / 1e9:.4g} GB")
    print(f"Banded array alone: {ab.nbytes / 1e6:.4g} MB")
    print(f"End displacement: {u[-1]:.8g} Å")
    return ab, f, u


@app.cell
def _(F, N, ab, f, k, np, u):
    reference = np.arange(1, N + 1) * F / k
    # Apply K without creating a dense N-by-N array.
    residual = ab[1] * u - f
    residual[:-1] += ab[0, 1:] * u[1:]
    residual[1:] += ab[2, :-1] * u[:-1]
    print("Relative solution error:", np.linalg.norm(u - reference) / np.linalg.norm(reference))
    print("Relative residual:", np.linalg.norm(residual) / np.linalg.norm(f))
    if N <= 100:
        dense_K = np.diag(ab[1]) + np.diag(ab[0, 1:], 1) + np.diag(ab[2, :-1], -1)
        print("Dense/banded agreement:", np.allclose(u, np.linalg.solve(dense_K, f)))
    return (reference,)


@app.cell
def _(N, np, plt, reference, u):
    _indices = np.unique(np.linspace(0, N - 1, min(N, 400), dtype=int))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(_indices + 1, reference[_indices], label="i F / k")
    ax.plot(_indices + 1, u[_indices], "--", label="Banded solve")
    ax.set(xlabel="Moving atom i", ylabel="Displacement (Å)")
    ax.legend()
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    Banded storage preserves the chain's local connections. It changes
    memory and computational cost, while the force-balance equations
    stay the same. Longer chains also become more sensitive: compare
    the analytical error as well as the residual when increasing `N`.
    For a direct dense comparison, set `N = 10`.
    """)
    return


if __name__ == "__main__":
    app.run()

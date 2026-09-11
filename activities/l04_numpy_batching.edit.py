# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import math
    import random
    from time import perf_counter
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return math, mo, np, perf_counter, plt, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Which batch size makes a large Buffon calculation fastest?**
    We use $L=D=1$, count crossings in batches, and estimate $\pi=2N/C$.
    Predict what happens if a batch is too small, then compare sizes at fixed $N$.

    The batched run is measured at the selected $N$. Reference times are
    extrapolated from at most $10^5$ Python throws and $10^6$ NumPy throws.
    A billion throws may take minutes; start with the default.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    settings = mo.ui.dictionary({
        "N": mo.ui.slider(
            steps=[10**k for k in range(3, 10)], value=1_000_000,
            label="Throws, N", show_value=True,
        ),
        "batch_size": mo.ui.slider(
            steps=[1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000],
            value=100_000, label="Batch size", show_value=True,
        ),
    }).form(submit_button_label="Measure")
    settings
    return (settings,)


@app.cell
def _(math, np, random, settings):
    settings.value
    rng = random.Random(374)
    rng_np = np.random.default_rng(374)

    def throw_needle(L: float, D: float) -> int:
        distance = 0.5 * D * rng.random()
        angle = 0.5 * math.pi * rng.random()
        return int(distance <= 0.5 * L * math.sin(angle))

    def buffon_experiment(N: int, L: float, D: float):
        crossings = 0
        for i in range(N):
            crossings += throw_needle(L, D)
        return 2 * L / D * N / crossings

    def count_crossings_np(chunk: int, L: float, D: float) -> int:
        d = 0.5 * D * rng_np.random(chunk)
        theta = 0.5 * np.pi * rng_np.random(chunk)
        return int(np.count_nonzero(d <= 0.5 * L * np.sin(theta)))

    def buffon_experiment_np(N: int, L: float, D: float):
        crossings = count_crossings_np(N, L, D)
        return 2 * L / D * N / crossings

    def buffon_experiment_np_batched(
        N: int, L: float, D: float, batch_size: int = 1_000_000
    ):
        crossings = 0
        for start in range(0, N, batch_size):
            chunk = min(batch_size, N - start)
            crossings += count_crossings_np(chunk, L, D)
        return 2 * L / D * N / crossings

    return buffon_experiment, buffon_experiment_np, buffon_experiment_np_batched


@app.cell(hide_code=True)
def _(
    buffon_experiment, buffon_experiment_np, buffon_experiment_np_batched,
    mo, perf_counter, settings,
):
    mo.stop(settings.value is None and mo.app_meta().mode != "script",
            mo.md("Choose N and a batch size, then press **Measure**."))
    config = settings.value or {"N": 1_000_000, "batch_size": 100_000}
    N = config["N"]
    batch_size = min(N, config["batch_size"])
    n_python = min(N, 100_000)
    n_numpy = min(N, 1_000_000)

    start = perf_counter()
    buffon_experiment(n_python, 1.0, 1.0)
    t_python_ref = perf_counter() - start
    start = perf_counter()
    buffon_experiment_np(n_numpy, 1.0, 1.0)
    t_numpy_ref = perf_counter() - start
    start = perf_counter()
    estimate = buffon_experiment_np_batched(N, 1.0, 1.0, batch_size)
    t_batch = perf_counter() - start
    return N, batch_size, estimate, n_numpy, n_python, t_batch, t_numpy_ref, t_python_ref


@app.cell(hide_code=True)
def _(
    N, batch_size, estimate, mo, n_numpy, n_python, np, plt,
    t_batch, t_numpy_ref, t_python_ref,
):
    labels = [
        "Python\n" + ("extrapolated" if N > n_python else "measured"),
        "Full-array NumPy\n" + ("extrapolated" if N > n_numpy else "measured"),
        "Batched NumPy\nmeasured",
    ]
    times = [t_python_ref * N / n_python, t_numpy_ref * N / n_numpy, t_batch]
    fig, ax = plt.subplots(figsize=(7, 3.2), layout="constrained")
    bars = ax.bar(labels, times, color=["#999999", "#a6bddb", "#2171b5"])
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title(f"{N:,} throws; batch size {batch_size:,}")
    ax.bar_label(bars, labels=[f"{t:.3g} s" for t in times], padding=4)
    ax.set_ylim(min(times) / 2, max(times) * 4)
    sigma = np.sqrt(np.pi**3 * (1 - 2 / np.pi) / (2 * N))
    mo.vstack([
        fig,
        mo.md(f"""
        **π estimate: {estimate:.6f}**; expected sampling uncertainty: {sigma:.2g} (1σ).
        The position and angle arrays together use **{16 * batch_size / 1e6:g} MB**
        per batch, plus temporaries.

        Reference measurements: Python {t_python_ref:.4f} s at {n_python:,} throws;
        full-array NumPy {t_numpy_ref:.4f} s at {n_numpy:,} throws.
        Extrapolation assumes constant time per throw and enough memory.
        Full-array NumPy would need **{16 * N / 1e9:g} GB** for positions and angles
        alone at the selected N.

        Keep N fixed and change the batch size. Which size gives the shortest runtime?
        """),
    ])
    return


if __name__ == "__main__":
    app.run()

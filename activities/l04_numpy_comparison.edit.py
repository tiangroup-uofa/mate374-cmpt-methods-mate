# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy"]
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
    return math, mo, np, perf_counter, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For Buffon's needles with $L=D=1$, the crossing probability is $2/\pi$.
    **Predict:** will NumPy's speedup grow or level off as $N$ increases?

    Runs at $10^8$ need several GB of memory and may fail in a browser.
    Start small; reload the notebook if a large allocation fails.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    settings = mo.ui.dictionary({
        "N": mo.ui.slider(
            steps=[10**k for k in range(3, 9)], value=100_000,
            label="Throws, N", show_value=True,
        ),
    }).form(submit_button_label="Measure")
    settings
    return (settings,)


@app.cell
def _(math, np, random, settings):
    # Fresh generators for each submitted experiment, outside the timed calls.
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

    def buffon_experiment_np(N: int, L: float, D: float):
        d = 0.5 * D * rng_np.random(N)
        theta = 0.5 * np.pi * rng_np.random(N)
        crossings = np.count_nonzero(d <= 0.5 * L * np.sin(theta))
        return 2 * L / D * N / crossings

    return buffon_experiment, buffon_experiment_np


@app.cell(hide_code=True)
def _(math, np):
    # Check identical inputs, including crossings and misses.
    test_d = np.array([0.0, 0.1, 0.4, 0.5])
    test_theta = np.array([0.0, 0.7, 0.2, np.pi / 2])
    scalar_decisions = [
        d <= 0.5 * math.sin(angle) for d, angle in zip(test_d, test_theta)
    ]
    assert np.array_equal(scalar_decisions, test_d <= 0.5 * np.sin(test_theta))
    return


@app.cell(hide_code=True)
def _(buffon_experiment, buffon_experiment_np, mo, perf_counter, settings):
    mo.stop(settings.value is None and mo.app_meta().mode != "script",
            mo.md("Choose N, then press **Measure**."))
    N = 100_000 if settings.value is None else settings.value["N"]
    start = perf_counter()
    pi_python = buffon_experiment(N, 1.0, 1.0)
    t_python = perf_counter() - start
    start = perf_counter()
    pi_numpy = buffon_experiment_np(N, 1.0, 1.0)
    t_numpy = perf_counter() - start
    return N, pi_numpy, pi_python, t_numpy, t_python


@app.cell(hide_code=True)
def _(N, mo, np, pi_numpy, pi_python, t_numpy, t_python):
    sigma = np.sqrt(np.pi**3 * (1 - 2 / np.pi) / (2 * N))
    mo.md(f"""
    | Method | Measured time (s) | Estimate of π |
    |---|---:|---:|
    | Python loop | {t_python:.4f} | {pi_python:.6f} |
    | NumPy arrays | {t_numpy:.4f} | {pi_numpy:.6f} |

    **Speedup: {t_python / t_numpy:.1f}×** for {N:,} throws.

    Expected sampling uncertainty (one standard deviation): **{sigma:.2g}**.
    Compare each estimate with π = {np.pi:.6f}; random runs will differ.
    Repeat the measurement to see how stable the timing is.
    """)
    return


if __name__ == "__main__":
    app.run()

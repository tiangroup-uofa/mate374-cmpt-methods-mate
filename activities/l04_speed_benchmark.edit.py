# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0"]
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

    return math, mo, perf_counter, random


@app.cell(hide_code=True)
def _(N, math, random):
    # Initialize outside both functions, before the timed call.
    rng = random.Random(374)

    def throw_needle(L: float, D: float) -> int:
        distance = 0.5 * D * rng.random()
        angle = 0.5 * math.pi * rng.random()
        return int(distance <= 0.5 * L * math.sin(angle))

    def buffon_experiment(N: int, L: float, D: float):
        crossings = 0
        for i in range(N):
            crossings += throw_needle(L, D)
        estimate = 2 * L / D * N / crossings
        return estimate

    return buffon_experiment, throw_needle


@app.cell(hide_code=True)
def _(mo):
    number_of_needles = mo.ui.slider(
        steps=[1_000, 10_000, 100_000, 500_000, 1_000_000],
        value=100_000,
        label="Number of throws, N",
        show_value=True,
        debounce=True,
    )
    number_of_needles
    return (number_of_needles,)


@app.cell(hide_code=True)
def _(number_of_needles):
    N = number_of_needles.value
    L = D = 1.0
    return D, L, N


@app.cell
def _(D, L, N, buffon_experiment, perf_counter):
    start = perf_counter()
    estimate = buffon_experiment(N, L, D)
    elapsed = perf_counter() - start
    return elapsed, estimate


@app.cell(hide_code=True)
def _(N, elapsed, mo):
    mo.md(f"""
    **{N:,} throws** · total time **{elapsed:.4f} s** ·
    average time per throw **{1e6 * elapsed / N:.3f} µs**
    """)
    return


if __name__ == "__main__":
    app.run()

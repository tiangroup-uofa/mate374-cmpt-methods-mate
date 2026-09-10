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
def _(math, random):
    def buffon_python(N):
        rng = random.Random(374)
        crossings = 0
        for _ in range(N):
            d = 0.5 * rng.random()
            theta = 0.5 * math.pi * rng.random()
            if d <= 0.5 * math.sin(theta):
                crossings += 1
        return crossings

    return (buffon_python,)


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
    return (N,)


@app.cell
def _(N, buffon_python, perf_counter):
    start = perf_counter()
    crossings = buffon_python(N)
    elapsed = perf_counter() - start
    return crossings, elapsed


@app.cell(hide_code=True)
def _(N, elapsed, mo):
    mo.md(f"""
    **{N:,} throws** · total time **{elapsed:.4f} s** ·
    average time per throw **{1e6 * elapsed / N:.3f} µs**
    """)
    return


if __name__ == "__main__":
    app.run()

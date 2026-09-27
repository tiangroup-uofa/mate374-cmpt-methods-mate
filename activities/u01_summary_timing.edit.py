# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    import numpy as np
    return np, time


@app.cell
def _(np, time):
    # Change N and rerun. The timed work includes array creation.
    N = 100_000

    def your_function(n):
        values = np.arange(n, dtype=np.float64)
        return np.sum(values**2)

    start = time.perf_counter()
    result = your_function(N)
    end = time.perf_counter()
    elapsed = end - start

    print(f"Sum of squares: {result:.6g}")
    print(f"Elapsed time: {elapsed:.6g} seconds")
    return


if __name__ == "__main__":
    app.run()

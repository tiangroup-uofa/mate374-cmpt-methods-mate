# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return (np, plt)


@app.cell
def _(np, plt):
    x = np.arange(8.0)
    y = np.array([1.15, 1.6, 3.45, 5.1, 8.0, 10.75, 14.9, 19.05])

    deg = 7   # try 1, 2, 3, 7

    poly = np.polynomial.Polynomial.fit(x, y, deg=deg)
    sse = np.sum((poly(x) - y)**2)

    x_fine = np.linspace(0.0, 7.0, 400)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, y, "o", label="data")
    ax.plot(x_fine, poly(x_fine), label=f"degree {deg}, SSE = {sse:.2e}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Mock final · Q4")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(np):
    def x(t):
        # Implement x(t).
        return np.full_like(t, np.nan, dtype=float)

    def y(t):
        # Implement y(t).
        return np.full_like(t, np.nan, dtype=float)

    return x, y


@app.cell
def _(np, plt, x, y):
    t_plot = np.linspace(-4.0, 4.0, 600)
    fig, ax = plt.subplots()
    ax.plot(x(t_plot), y(t_plot))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig
    return


if __name__ == "__main__":
    app.run()

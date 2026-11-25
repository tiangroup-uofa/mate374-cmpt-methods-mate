# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Mock final · Q4 · completed")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mock final · Q4 · Length of a pretzel

    $x(t)=(2.5-0.3t^2)\cos t$, $y(t)=(3.3-0.4t^2)\sin t$, $-4\leq t\leq3$.

    Save your edited notebook when finished.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (a) The curve · 5 points

    Write Python functions for x(t) and y(t). Run the supplied plotting cell to display the pretzel's shape.
    """)
    return


@app.cell
def _(np):
    def x(t):
        return (2.5 - 0.3 * t**2) * np.cos(t)

    def y(t):
        return (3.3 - 0.4 * t**2) * np.sin(t)

    return x, y


@app.cell
def _(np, plt, x, y):
    t_plot = np.linspace(-4.0, 3.0, 600)
    curve_fig, curve_ax = plt.subplots(figsize=(4, 4))
    curve_ax.plot(x(t_plot), y(t_plot), color="#b8582d")
    curve_ax.set_aspect("equal")
    curve_ax.set_xlabel("x")
    curve_ax.set_ylabel("y")
    curve_fig.tight_layout()
    curve_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (b) Numerical integration · 20 points

    Determine the length by numerical integration, using a method of your choice. Report the length, identify your method, and explain why it is suitable for this problem. The reference length is 17.2750.
    """)
    return


@app.cell
def _(np, x, y):
    t_int = np.linspace(-4.0, 3.0, 10001)
    dxdt = np.gradient(x(t_int), t_int)
    dydt = np.gradient(y(t_int), t_int)
    speed = np.sqrt(dxdt**2 + dydt**2)
    length = np.trapezoid(speed, t_int)
    print(f"Length: {length:.10f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This example uses finite differences for the coordinate derivatives and the composite trapezoidal rule for the speed integral. The fine uniform grid resolves the smooth curve. The result rounds to the reference length, 17.2750.
    """)
    return


if __name__ == "__main__":
    app.run()

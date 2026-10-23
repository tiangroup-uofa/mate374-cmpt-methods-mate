# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _():
    def forward(f, x, h):
        return (f(x + h) - f(x)) / h


    def central(f, x, h):
        return (f(x + h) - f(x - h)) / (2 * h)

    return central, forward


@app.cell
def _(mo):
    x_choice = mo.ui.dropdown(options=["0.5", "1.0", "2.0"], value="1.0", label="x")
    precision = mo.ui.dropdown(options=["float64", "float32"], value="float64", label="precision")
    return precision, x_choice


@app.cell
def _(central, forward, np, precision, x_choice):
    dtype = np.dtype(precision.value).type
    x = dtype(float(x_choice.value))
    h = np.logspace(-16, 0, 161).astype(dtype)
    exact = np.cos(np.float64(x))

    err_forward = np.abs(forward(np.sin, x, h).astype(np.float64) - exact)
    err_central = np.abs(central(np.sin, x, h).astype(np.float64) - exact)

    rows = ["| $h$ | forward error | central error |", "|---|---|---|"]
    for _value in [1e-1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]:
        _h = dtype(_value)
        _ef = abs(float(forward(np.sin, x, _h)) - exact)
        _ec = abs(float(central(np.sin, x, _h)) - exact)
        rows.append(f"| {_value:.0e} | {_ef:.2e} | {_ec:.2e} |")
    table = "\n".join(rows)
    return dtype, err_central, err_forward, h, table, x


@app.cell
def _(dtype, err_central, err_forward, h, mo, np, plt, precision, table, x, x_choice):
    eps = np.finfo(dtype).eps
    hh = h.astype(np.float64)
    xx = float(x)
    roundoff = eps * abs(np.sin(xx)) / hh
    model_forward = abs(np.sin(xx)) / 2 * hh + 2 * roundoff
    model_central = abs(np.cos(xx)) / 6 * hh**2 + roundoff

    fig, ax = plt.subplots(figsize=(6.2, 3.8), layout="constrained")
    # Exact zeros cannot be drawn on a logarithmic axis.
    ax.loglog(1 / hh, np.where(err_forward > 0, err_forward, np.nan), color="C0", label="forward")
    ax.loglog(1 / hh, np.where(err_central > 0, err_central, np.nan), color="C3", label="central")
    ax.loglog(1 / hh, model_forward, ":", color="C0", label="model  Ah + B/h")
    ax.loglog(1 / hh, model_central, ":", color="C3", label="model  Ah² + B/h")
    ax.set(xlabel="1/h", ylabel="absolute error in f'(x)", ylim=(1e-14, 1e1))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    mo.vstack(
        [
            mo.hstack([x_choice, precision, mo.md(f"machine epsilon $\\epsilon$ = {eps:.2e}")], justify="start", align="center", gap=3),
            mo.hstack([fig, mo.md(table)], justify="start", align="center", gap=2, widths=[3, 2]),
        ]
    )
    return



if __name__ == "__main__":
    app.run()

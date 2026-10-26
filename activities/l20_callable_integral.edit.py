# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import integrate

    return integrate, mo, np


@app.cell
def _():
    def f(x):
        return x + x**3

    return (f,)


@app.cell
def _(f, integrate):
    value, error_estimate = integrate.quad(f, 0.0, 2.0)
    return error_estimate, value


@app.cell
def _(error_estimate, mo, value):
    mo.md(f"""
    **Integral:** {value:.10f}

    **Estimated absolute error:** {error_estimate:.2e}

    **Exact value:** 6
    """)
    return


if __name__ == "__main__":
    app.run()

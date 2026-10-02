# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _():
    # M and x_true
    return


@app.cell
def _():
    # b = M @ x_true
    return


@app.cell
def _():
    # inv(M) @ b and solve(M, b)
    return


@app.cell
def _():
    # residuals
    return


@app.cell
def _():
    # det, cond, inv(M)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

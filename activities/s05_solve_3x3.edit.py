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
    # A and b
    return


@app.cell
def _():
    # inv(A) @ b and solve(A, b)
    return


@app.cell
def _():
    # residual, relative residual, and condition number
    return


if __name__ == "__main__":
    app.run()

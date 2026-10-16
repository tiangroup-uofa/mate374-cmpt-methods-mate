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
def _(np):
    # Intended: the squares 1, 4, 9, 16
    x = np.array([1, 2, 3, 4])
    x ^ 2
    return


if __name__ == "__main__":
    app.run()

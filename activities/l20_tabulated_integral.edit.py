# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from numpy import trapezoid
    from scipy.integrate import simpson

    x = [0.0, 0.5, 1.0, 1.5, 2.0]
    y = [0.0, 0.625, 2.0, 4.875, 10.0]

    trapezoid(y, x), simpson(y, x=x)
    return


if __name__ == "__main__":
    app.run()

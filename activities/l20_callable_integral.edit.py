# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from scipy.integrate import quad


    def f(x):
        return x + x**3


    value, error_estimate = quad(f, 0, 2)
    value, error_estimate
    return


if __name__ == "__main__":
    app.run()

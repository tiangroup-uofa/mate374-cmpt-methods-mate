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
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])        # h
    c = np.array([2.0, 1.2, 0.75, 0.42, 0.26])     # mol/L

    # Model c = c0 exp(-k t), linearized as ln c = ln c0 - k t
    line = np.polynomial.Polynomial.fit(t, np.log(c), deg=1)
    c0, minus_k = line.coef
    k = -minus_k

    print("c0 =", c0, " k =", k)
    print("predicted:", c0 * np.exp(-k * t))
    print("measured: ", c)
    return


if __name__ == "__main__":
    app.run()

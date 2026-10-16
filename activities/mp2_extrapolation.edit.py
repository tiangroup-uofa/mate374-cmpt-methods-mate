# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import scipy as sp
    return (np, sp)


@app.cell
def _(np, sp):
    v = np.array([12.0, 20.0, 28.0, 36.0, 44.0])       # wind speed, mph
    P = np.array([320.0, 490.0, 540.0, 500.0, 480.0])  # power, W

    spline = sp.interpolate.CubicSpline(v, P)
    pchip = sp.interpolate.PchipInterpolator(v, P)

    # Intended: the power at 60 mph
    print("CubicSpline at 60 mph:", spline(60.0))
    print("PCHIP at 60 mph:      ", pchip(60.0))
    return


if __name__ == "__main__":
    app.run()

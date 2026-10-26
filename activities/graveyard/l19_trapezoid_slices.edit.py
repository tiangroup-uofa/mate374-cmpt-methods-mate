# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md("""
    ### Displacement from a velocity record

    Predict the displacement, then run the slices below. Edit the velocities to see how the total changes.
    """)
    return


@app.cell
def _(np):
    x = np.array([0.0, 1.0, 3.0, 4.0])  # time (ps)
    y = np.array([0.0, 2.0, 2.0, 0.0])  # velocity (Å/ps)
    return x, y


@app.cell
def _(np, x, y):
    widths = x[1:] - x[:-1]
    average_velocities = (y[:-1] + y[1:]) / 2
    displacement = np.sum(widths * average_velocities)
    numpy_displacement = np.trapezoid(y, x)
    return average_velocities, displacement, numpy_displacement, widths


@app.cell
def _(average_velocities, displacement, mo, numpy_displacement, widths):
    mo.md(f"""
    - Step widths: **{widths} ps**
    - Average velocities: **{average_velocities} Å/ps**
    - Displacement from slices: **{displacement:g} Å**
    - Displacement from `np.trapezoid(y, x)`: **{numpy_displacement:g} Å**
    """)
    return


if __name__ == "__main__":
    app.run()

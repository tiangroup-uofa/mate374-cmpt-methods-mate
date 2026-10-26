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
def _(np):
    # Illustrative engineering strain and stress (MPa) for a structural steel, up to fracture
    strain = np.array([0, 0.0006, 0.0012, 0.005, 0.015, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.24, 0.27, 0.30])
    stress = np.array([0, 125, 250, 250, 250, 288, 325, 362, 390, 405, 412, 413, 409, 401])
    return strain, stress


@app.cell
def _(mo):
    keep = mo.ui.dropdown(
        options={"all 14 points": 1, "every 2nd point": 2, "every 3rd point": 3},
        value="all 14 points",
        label="points used",
    )
    return (keep,)


@app.cell
def _(keep, mo, np, strain, stress):
    idx = sorted(set(range(0, strain.size, keep.value)) | {strain.size - 1})   # always keep the fracture point
    x, y = strain[idx], stress[idx]

    # Composite trapezoidal rule written with slices: one trapezoid per subinterval
    h = x[1:] - x[:-1]                         # widths h_i, not all equal
    U_formula = np.sum(h * (y[:-1] + y[1:]) / 2)

    # The same calculation in NumPy
    U_numpy = np.trapezoid(y, x)

    mo.vstack(
        [
            keep,
            mo.md(
                f"""
    | calculation | toughness (MJ/m³) |
    |---|---|
    | sum of trapezoids with slices | {U_formula:.3f} |
    | `np.trapezoid(y, x)` | {U_numpy:.3f} |
    """
            ),
        ]
    )
    return x, y


@app.cell
def _(plt, strain, stress, x, y):
    fig, ax = plt.subplots(figsize=(7.5, 3.4), layout="constrained")
    for _i in range(x.size - 1):
        ax.fill_between(x[_i:_i + 2], 0, y[_i:_i + 2], color="C2", alpha=0.25 + 0.15 * (_i % 2), lw=0)
    ax.plot(x, y, "o-", color="k", ms=4, lw=1.2)
    ax.plot(strain[-1], stress[-1], "x", color="C3", ms=10, mew=2)
    ax.set(xlabel="engineering strain ε", ylabel="stress σ (MPa)", ylim=(0, 450), title=f"{x.size} points, {x.size - 1} trapezoids")
    ax.grid(alpha=0.25)
    fig
    return


if __name__ == "__main__":
    app.run()

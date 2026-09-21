# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline, PchipInterpolator
    return CubicSpline, PchipInterpolator, mo, np, plt


@app.cell
def data(np):
    # Synthetic property table: temperature (K), thermal conductivity (W m⁻¹ K⁻¹).
    T = np.array([300., 400., 500., 600., 700.])
    conductivity = np.array([15., 16., 16.2, 19., 20.])
    query_T = 450.  # Try 750 K, outside the table.
    return T, conductivity, query_T


@app.cell
def interpolate(CubicSpline, PchipInterpolator, T, conductivity, np, query_T):
    # Change bc_type to "not-a-knot", or extrapolate to True, then compare.
    cubic = CubicSpline(T, conductivity, bc_type="natural", extrapolate=False)
    pchip = PchipInterpolator(T, conductivity, extrapolate=False)
    linear_value = np.interp(query_T, T, conductivity, left=np.nan, right=np.nan)
    print(f"At {query_T:g} K (W m⁻¹ K⁻¹):")
    print(f"linear = {linear_value:.4f}; cubic = {cubic(query_T):.4f}; PCHIP = {pchip(query_T):.4f}")
    print("Cubic node residual:", np.max(np.abs(cubic(T) - conductivity)))
    print("Cubic coefficients shape:", cubic.c.shape)
    return cubic, pchip


@app.cell(hide_code=True)
def plot(T, conductivity, cubic, np, pchip, plt, query_T):
    _grid = np.linspace(250, 750, 600)
    spline_figure, _ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    _ax.plot(_grid, np.interp(_grid, T, conductivity, left=np.nan, right=np.nan), label="Piecewise linear")
    _ax.plot(_grid, cubic(_grid), label="Cubic spline")
    _ax.plot(_grid, pchip(_grid), ls="--", label="PCHIP")
    _ax.scatter(T, conductivity, color="black", zorder=4, label="Tabulated data")
    _ax.axvline(query_T, color="0.4", ls=":", label="Query temperature")
    _ax.axvspan(250, T[0], color="0.5", alpha=0.1)
    _ax.axvspan(T[-1], 750, color="0.5", alpha=0.1)
    _ax.set(xlabel="Temperature (K)", ylabel="Thermal conductivity (W m⁻¹ K⁻¹)", title="Does the interpolant preserve the increasing trend?")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.2)
    spline_figure
    return (spline_figure,)


if __name__ == "__main__":
    app.run()

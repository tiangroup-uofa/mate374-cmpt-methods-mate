# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.polynomial import Polynomial
    return Polynomial, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    These synthetic data show the strain $\varepsilon$ (mm/m) and stress $\sigma$
    (MPa) of a material. We want to fit stress as a linear function of strain:

    $$\hat f(\varepsilon)=a_0+a_1\varepsilon.$$

    Here $a_0$ is the stress intercept and $a_1$ is the slope.
    **How well does the line fit the data?** Compare the fitted curve with the
    measurements and inspect the residuals.
    """)
    return


@app.cell
def data(np):
    # Synthetic elastic tensile data; x is strain expressed in mm/m.
    strain = np.array([0., 0.5, 1., 1.5, 2., 2.5, 3.])
    stress = np.array([2., 33., 73., 102., 143., 172., 214.])  # MPa
    return strain, stress


@app.cell
def fit_data(Polynomial, np, strain, stress):
    degree = 1  # Try 2, then 6. What happens between the measurements?
    fitted = Polynomial.fit(strain, stress, deg=degree)
    residuals = stress - fitted(strain)
    SSE = np.sum(residuals**2)
    RMSE = np.sqrt(np.mean(residuals**2))
    R_squared = 1 - SSE / np.sum((stress - stress.mean())**2)
    print("Coefficients [a0, a1, ...] in original x units:", fitted.convert().coef)
    print(f"SSE = {SSE:.3f} MPa²; RMSE = {RMSE:.3f} MPa; R² = {R_squared:.6f}")
    if degree == 1:
        print(f"Estimated Young's modulus = {fitted.convert().coef[1]:.3f} GPa")
    return RMSE, R_squared, SSE, degree, fitted, residuals


@app.cell(hide_code=True)
def plot(degree, fitted, np, plt, residuals, strain, stress):
    regression_figure, _axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    _grid = np.linspace(-0.2, 3.2, 400)
    _axes[0].scatter(strain, stress, color="black", label="Synthetic measurements", zorder=4)
    _axes[0].plot(_grid, fitted(_grid), label=f"Degree {degree} least-squares fit")
    _axes[0].vlines(strain, fitted(strain), stress, color="tab:red", label="Residuals")
    _axes[0].set(xlabel="Strain (mm/m)", ylabel="Stress (MPa)")
    _axes[0].legend(fontsize=8)
    _axes[1].axhline(0, color="0.4", lw=0.8)
    _axes[1].stem(strain, residuals, basefmt=" ")
    _axes[1].set(xlabel="Strain (mm/m)", ylabel="Measured − fitted stress (MPa)", title="Residual check")
    for _ax in _axes:
        _ax.grid(alpha=0.2)
    regression_figure
    return (regression_figure,)


if __name__ == "__main__":
    app.run()

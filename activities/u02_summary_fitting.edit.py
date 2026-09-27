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
    from numpy.polynomial import Polynomial
    from scipy.optimize import curve_fit, minimize
    return Polynomial, curve_fit, minimize, np


@app.cell
def _(np):
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.array([1.1, 1.7, 3.2, 4.8, 7.5])
    return x_data, y_data


@app.cell
def _(Polynomial, np, x_data, y_data):
    # Use degree 1 polynomial (linear fitting)
    line = Polynomial.fit(x_data, y_data, deg=1)
    print("Line: intercept, slope =", line.convert().coef)
    print("Residuals:", line(x_data) - y_data)
    print("Sum of squares:", np.sum((line(x_data) - y_data)**2))
    return


@app.cell
def _(curve_fit, minimize, np, x_data, y_data):
    def model(x, amplitude, rate):
        """General non-linear model, 2 parameters"""
        return amplitude * np.exp(rate * x)

    def objective(parameters):
        residuals = model(x_data, *parameters) - y_data
        return np.sum(residuals**2)

    initial_parameters = [1.0, 0.5]

    # Use curve_fit, just supply the function to fit itself!
    parameters, covariance = curve_fit(model, x_data, y_data, p0=initial_parameters)
    print("curve_fit [amplitude, rate]:", parameters)


    # Use general minimize function: use objective (residual)
    # instead of direct fitting
    fit = minimize(objective, x0=initial_parameters)

    print("minimize [amplitude, rate]:", fit.x, "success:", fit.success)
    return


if __name__ == "__main__":
    app.run()

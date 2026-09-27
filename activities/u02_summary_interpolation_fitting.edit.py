# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.polynomial import Polynomial
    from scipy.interpolate import lagrange, CubicSpline, PchipInterpolator
    from scipy.optimize import curve_fit, minimize
    return CubicSpline, PchipInterpolator, Polynomial, curve_fit, lagrange, minimize, np, plt


@app.cell
def _(np):
    # Synthetic dimensionless data. Keep the x values distinct and increasing.
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.array([1.1, 1.7, 3.2, 4.8, 7.5])
    x_query = 1.5
    return x_data, x_query, y_data


@app.cell
def _(CubicSpline, PchipInterpolator, Polynomial, lagrange, np, x_data, x_query, y_data):
    p = Polynomial.fit(x_data, y_data, deg=len(x_data) - 1)
    p_lagrange = lagrange(x_data, y_data)
    y_linear = np.interp(x_query, x_data, y_data)
    cubic = CubicSpline(x_data, y_data)
    shape_preserving = PchipInterpolator(x_data, y_data)

    print(f"Estimates at x = {x_query}:")
    print("Global polynomial:", p(x_query))
    print("Lagrange polynomial:", p_lagrange(x_query))
    print("Piecewise linear:", y_linear)
    print("Cubic spline:", cubic(x_query))
    print("PCHIP:", shape_preserving(x_query))
    return cubic, p, shape_preserving


@app.cell
def _(Polynomial, curve_fit, minimize, np, x_data, y_data):
    line = Polynomial.fit(x_data, y_data, deg=1)

    def model(x, amplitude, rate):
        return amplitude * np.exp(rate * x)

    initial_parameters = [1.0, 0.5]
    parameters, covariance = curve_fit(model, x_data, y_data, p0=initial_parameters)

    def objective(parameters):
        residuals = model(x_data, *parameters) - y_data
        return np.sum(residuals**2)

    fit = minimize(objective, x0=initial_parameters)
    print("curve_fit [amplitude, rate]:", parameters)
    print("Sum of squared residuals:", objective(parameters))
    print("minimize [amplitude, rate]:", fit.x, "success:", fit.success)
    print("Sum of squared residuals:", fit.fun)
    return line, model, parameters


@app.cell
def _(cubic, line, model, np, p, parameters, plt, shape_preserving, x_data, y_data):
    x_plot = np.linspace(x_data[0], x_data[-1], 301)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x_data, y_data, color="black", label="Data", zorder=5)
    ax.plot(x_plot, p(x_plot), label="Global interpolation")
    ax.plot(x_plot, cubic(x_plot), label="Cubic spline")
    ax.plot(x_plot, shape_preserving(x_plot), label="PCHIP")
    ax.plot(x_plot, line(x_plot), "--", label="Least-squares line")
    ax.plot(x_plot, model(x_plot, *parameters), "--", label="Exponential fit")
    ax.set(xlabel="x (dimensionless)", ylabel="y (dimensionless)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()

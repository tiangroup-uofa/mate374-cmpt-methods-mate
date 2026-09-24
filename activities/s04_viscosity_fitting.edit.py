# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    return curve_fit, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## Fit bitumen viscosity

    The empirical model is $\mu=bT^m$, with `T` entered as the numerical
    temperature in °C and viscosity in Pa·s. The log-space fit uses the four
    sums from L09, followed by a direct fit with `curve_fit`.
    Working calculations are supplied in editable cells.

    **Predict:** which fit will have the smaller sum of squared viscosity
    residuals? Which will have the smaller sum of squared log residuals?
    """)
    return


@app.cell(hide_code=False)
def data(np):
    T = np.array([18., 33., 57., 70.])  # Numerical Celsius temperatures
    mu = np.array([1000., 100., 10., 2.])  # Pa s
    return T, mu


@app.cell(hide_code=False)
def linear_fit(T, mu, np):
    X = np.log(T)
    Y = np.log(mu)
    N = len(X)
    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sxx = np.sum(X**2)
    Sxy = np.sum(X * Y)
    m_log = (N * Sxy - Sx * Sy) / (N * Sxx - Sx**2)
    alpha_log = (Sy - m_log * Sx) / N
    b_log = np.exp(alpha_log)
    log_parameters = np.array([alpha_log, m_log])
    print("Sx, Sy, Sxx, Sxy:", Sx, Sy, Sxx, Sxy)
    print(f"alpha = {alpha_log:.8f}, m = {m_log:.8f}, b = {b_log:.8g}")
    return N, Sx, Sxx, Sxy, Sy, X, Y, alpha_log, b_log, log_parameters, m_log


@app.cell(hide_code=False)
def normal_equations(N, Sx, Sxx, Sxy, Sy, alpha_log, m_log):
    # Check the two equations used to derive the formulas.
    print("First normal-equation residual:", N * alpha_log + Sx * m_log - Sy)
    print("Second normal-equation residual:", Sx * alpha_log + Sxx * m_log - Sxy)
    return


@app.cell(hide_code=False)
def model(np):
    def viscosity(T, alpha, m):
        return np.exp(alpha + m * np.log(T))

    return (viscosity,)


@app.cell(hide_code=False)
def direct_fit(T, curve_fit, log_parameters, mu, np, viscosity):
    direct_parameters, covariance = curve_fit(
        viscosity, T, mu, p0=log_parameters, maxfev=10000
    )
    alpha_direct, m_direct = direct_parameters
    b_direct = np.exp(alpha_direct)
    print(f"alpha = {alpha_direct:.8f}, m = {m_direct:.8f}, b = {b_direct:.8g}")
    return b_direct, direct_parameters, m_direct


@app.cell(hide_code=False)
def residual_calculation(T, direct_parameters, log_parameters, mu, np, viscosity):
    predicted_log = viscosity(T, *log_parameters)
    predicted_direct = viscosity(T, *direct_parameters)
    residual_log = mu - predicted_log
    residual_direct = mu - predicted_direct
    print("Columns: T, measured, log-fit prediction, direct-fit prediction (Pa s)")
    print(np.column_stack([T, mu, predicted_log, predicted_direct]))
    print("Log-fit residuals (Pa s):", residual_log)
    print("Direct-fit residuals (Pa s):", residual_direct)
    print("Direct-fit relative residuals (%):", 100 * residual_direct / mu)
    return predicted_direct, predicted_log, residual_direct, residual_log


@app.cell(hide_code=False)
def error_measures(mu, np, predicted_direct, predicted_log, residual_direct, residual_log):
    sse_log_fit = np.sum(residual_log**2)
    sse_direct_fit = np.sum(residual_direct**2)
    log_sse_log_fit = np.sum((np.log(mu) - np.log(predicted_log))**2)
    log_sse_direct_fit = np.sum((np.log(mu) - np.log(predicted_direct))**2)
    print("Viscosity SSE (log fit, direct fit):", sse_log_fit, sse_direct_fit)
    print("Viscosity RMSE (log fit, direct fit):",
          np.sqrt(sse_log_fit / len(mu)), np.sqrt(sse_direct_fit / len(mu)))
    print("Log-space SSE (log fit, direct fit):", log_sse_log_fit, log_sse_direct_fit)
    return log_sse_direct_fit, log_sse_log_fit, sse_direct_fit, sse_log_fit


@app.cell(hide_code=False)
def prediction(direct_parameters, log_parameters, viscosity):
    T_new = 45.0  # Change this numerical Celsius temperature.
    print(f"At {T_new:g} °C, viscosity predictions (Pa s):")
    print("Log fit:", viscosity(T_new, *log_parameters))
    print("Direct fit:", viscosity(T_new, *direct_parameters))
    return (T_new,)


@app.cell(hide_code=True)
def plot(T, direct_parameters, log_parameters, mu, np, plt, residual_direct, residual_log, viscosity):
    def plot_fits():
        grid = np.linspace(T.min(), T.max(), 300)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].semilogy(T, mu, "ko", label="Measurements")
        axes[0].semilogy(grid, viscosity(grid, *log_parameters), label="Log-space fit")
        axes[0].semilogy(grid, viscosity(grid, *direct_parameters), "--", label="Direct fit")
        axes[0].set(xlabel="Temperature (°C)", ylabel="Viscosity (Pa s)", title="Same power law, different objectives")
        axes[1].plot(T, 100 * residual_log / mu, "o-", label="Log-space fit")
        axes[1].plot(T, 100 * residual_direct / mu, "s--", label="Direct fit")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set(xlabel="Temperature (°C)", ylabel="100 × (measured − fitted) / measured (%)",
                    title="Relative residuals at the measured points")
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    fitting_figure = plot_fits()
    fitting_figure
    return (fitting_figure,)


@app.cell(hide_code=True)
def follow_up(mo):
    mo.md(r"""
    At 70 °C, is the direct-fit residual small relative to the measured viscosity?
    Explain why a small overall RMSE can coexist with a large percentage mismatch.

    **Your calculation:** use the prediction cell to compare both fits at
    45 °C. Which objective gives more influence to the 1000 Pa·s measurement?
    """)
    return


if __name__ == "__main__":
    app.run()

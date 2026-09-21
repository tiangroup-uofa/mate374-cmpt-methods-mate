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
    temperature in °C and viscosity in Pa·s. Type the log-space fit and the
    direct-fit calls into their existing cells. A working reference is supplied.

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
    design = np.column_stack([np.ones_like(X), X])
    log_parameters, _, rank, _ = np.linalg.lstsq(design, Y, rcond=None)
    alpha_log, m_log = log_parameters
    b_log = np.exp(alpha_log)
    print("Design matrix shape:", design.shape, "rank:", rank)
    print(f"alpha = {alpha_log:.8f}, m = {m_log:.8f}, b = {b_log:.8g}")
    return X, Y, alpha_log, b_log, design, log_parameters, m_log


@app.cell(hide_code=False)
def normal_equations(Y, design, np):
    normal_matrix = design.T @ design
    normal_rhs = design.T @ Y
    normal_parameters = np.linalg.solve(normal_matrix, normal_rhs)
    print("Normal matrix:\n", normal_matrix)
    print("Right-hand side:", normal_rhs)
    print("[alpha, m] from the normal equations:", normal_parameters)
    return normal_matrix, normal_parameters, normal_rhs


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

    **Your code change:** add `sigma=mu` to `curve_fit` to minimize squared
    relative residuals. Compare the parameters and both error measures again,
    then remove that argument to restore the original unweighted calculation.
    """)
    return


if __name__ == "__main__":
    app.run()

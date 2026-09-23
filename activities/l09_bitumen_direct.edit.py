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
    from scipy.optimize import curve_fit, minimize
    return curve_fit, minimize, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## Bitumen viscosity: direct nonlinear least squares
    Fit $\hat f(T)=A(T/298)^{-p}$ to the same four synthetic series.
    **Prediction:** will `curve_fit` and `minimize` agree when they minimize the same error?
    Edit the starting values or arrays below. Temperatures are in K and viscosities in Pa·s.
    """)
    return


@app.cell(hide_code=True)
def data(np):
    # Rounded values from scripts/l09_intro_figures.py; no measured paper data.
    T = np.array([
        298.00000000, 299.11111111, 300.22222222, 301.33333333, 302.44444444, 303.55555556, 304.66666667, 305.77777778,
        306.88888889, 308.00000000, 309.11111111, 310.22222222, 311.33333333, 312.44444444, 313.55555556, 314.66666667,
        315.77777778, 316.88888889, 318.00000000, 319.11111111, 320.22222222, 321.33333333, 322.44444444, 323.55555556,
        324.66666667, 325.77777778, 326.88888889, 328.00000000, 329.11111111, 330.22222222, 331.33333333, 332.44444444,
        333.55555556, 334.66666667, 335.77777778, 336.88888889, 338.00000000, 339.11111111, 340.22222222, 341.33333333,
        342.44444444, 343.55555556, 344.66666667, 345.77777778, 346.88888889, 348.00000000, 349.11111111, 350.22222222,
        351.33333333, 352.44444444, 353.55555556, 354.66666667, 355.77777778, 356.88888889, 358.00000000, 359.11111111,
        360.22222222, 361.33333333, 362.44444444, 363.55555556, 364.66666667, 365.77777778, 366.88888889, 368.00000000,
    ])
    mu = np.array([
        109.08918943, 87.49324049, 87.60779425, 78.11148925, 93.51551536, 86.84546057, 72.80816584, 65.30338428,
        73.88708819, 93.56381721, 73.70966062, 66.14465183, 85.63139691, 60.14891629, 79.44549688, 48.98505769,
        77.99797006, 70.25684455, 46.91437708, 48.78851085, 64.94181869, 63.53870494, 40.33290502, 35.38402517,
        60.51365197, 54.09848162, 54.44509758, 46.12001805, 59.16328034, 60.42177372, 37.41920435, 48.58807083,
        39.39415875, 47.31505906, 31.73277989, 29.20288274, 49.23098261, 43.88442204, 30.70434637, 26.63848080,
        31.28256150, 35.38004833, 29.07260984, 28.39283583, 28.85740285, 33.99215738, 24.17886789, 25.27485005,
        25.01252142, 21.73304398, 24.26578919, 22.97028587, 22.36474206, 24.88448308, 22.20304664, 23.20927715,
        25.05306832, 27.47747626, 18.86610730, 21.38751028, 21.65384840, 21.38660428, 18.68652432, 19.27866727,
    ])
    return T, mu


@app.cell
def model():
    T_ref = 298.0  # K; A is the fitted viscosity here, in Pa·s.

    def power_law(temperature, A, p):
        return A * (temperature / T_ref)**(-p)

    return T_ref, power_law


@app.cell
def fit(T, T_ref, curve_fit, mu, power_law):
    parameters, covariance = curve_fit(
        power_law, T, mu, p0=[100.0, 8.0], bounds=(0, [float("inf"), 50.0])
    )
    A, p = parameters
    alpha = A * T_ref**p

    def f_hat(temperature):
        return power_law(temperature, A, p)

    print(f"A = {A:.6f} Pa·s, p = {p:.6f}")
    print(f"alpha = {alpha:.6g} (Pa·s K^p)")
    return A, alpha, covariance, f_hat, p, parameters


@app.cell
def residuals(T, f_hat, mu, np):
    residual = mu - f_hat(T)
    log_residual = np.log(mu) - np.log(f_hat(T))
    sse = np.sum(residual**2)
    log_sse = np.sum(log_residual**2)
    print(f"Viscosity SSE = {sse:.4f} (Pa·s)^2")
    print(f"Log-space SSE = {log_sse:.6f}")
    return log_residual, log_sse, residual, sse


@app.cell(hide_code=True)
def objective_intro(mo):
    mo.md("""
    ### The same fit with a scalar objective
    `curve_fit` constructs the squared-error sum from the model and data.
    For `minimize`, we supply that sum ourselves. Keep the same starting values
    and bounds, then compare the coefficients and convergence message.
    """)
    return


@app.cell
def general_optimizer(T, minimize, mu, np, power_law):
    def objective(coefficients):
        return np.sum((mu - power_law(T, *coefficients))**2)

    result = minimize(
        objective, x0=[100.0, 8.0], method="L-BFGS-B",
        bounds=[(0, None), (0, 50)],
    )
    print("Converged:", result.success, result.message)
    print("A, p:", result.x)
    print("Viscosity SSE:", result.fun)
    return objective, result


@app.cell(hide_code=True)
def plot(T, f_hat, mu, np, plt, residual):
    figure, _axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    _grid = np.linspace(T.min(), T.max(), 500)
    _axes[0].scatter(T, mu, s=18, label="Synthetic data")
    _axes[0].plot(_grid, f_hat(_grid), color="#29805c", label="Direct fit")
    _axes[0].set(xlabel="Temperature (K)", ylabel="Viscosity (Pa·s)")
    _axes[0].legend()
    _axes[1].scatter(T, residual, s=18)
    _axes[1].axhline(0, color="0.4", lw=1)
    _axes[1].set(xlabel="Temperature (K)", ylabel="Viscosity residual (Pa·s)")
    figure
    return (figure,)


@app.cell(hide_code=True)
def interpretation(mo):
    mo.md("""
    This fit minimizes the **viscosity-space SSE**. Compare its exponent and both
    error sums with the linearized fit. Try another positive starting guess:
    do `curve_fit` and `minimize` still reach the same coefficients? A convergence
    message describes the optimizer's stopping condition; the residual plot helps
    assess whether the power law describes the measurements.
    """)
    return


if __name__ == "__main__":
    app.run()

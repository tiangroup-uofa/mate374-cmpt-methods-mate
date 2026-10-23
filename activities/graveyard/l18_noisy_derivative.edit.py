# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.15", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.polynomial import Polynomial
    from scipy.interpolate import CubicSpline
    from scipy.differentiate import derivative
    return CubicSpline, Polynomial, derivative, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Can noisy energies give a reliable force?
    Near a bond minimum, take $E(q)=kq^2/2$ with $k=2$ eV/Å².
    We add independent synthetic energy noise, then compare a raw finite
    difference, an interpolating cubic spline, and a quadratic least-squares fit.
    **Predict:** will passing exactly through every energy improve the force?
    """)
    return


@app.cell
def _(np):
    noise_eV = 0.001                  # Try 0.0, 0.0001, 0.005
    n_samples = 41                   # Try 21 or 81 at fixed noise
    stiffness = 2.0                  # eV / Å²
    q = np.linspace(-0.2, 0.2, n_samples)  # Å
    rng = np.random.default_rng(374)
    clean_energy = 0.5 * stiffness * q**2
    measured_energy = clean_energy + rng.normal(0.0, noise_eV, q.size)
    return measured_energy, noise_eV, q, stiffness


@app.cell
def _(CubicSpline, Polynomial, measured_energy, np, q):
    raw_force = -np.gradient(measured_energy, q, edge_order=2)
    spline = CubicSpline(q, measured_energy)
    fit = Polynomial.fit(q, measured_energy, deg=2)
    spline_force = -spline.derivative()(q)
    fitted_force = -fit.deriv()(q)
    return fit, fitted_force, raw_force, spline, spline_force


@app.cell
def _(fit, measured_energy, np, plt, q, raw_force, spline, stiffness):
    fine_q = np.linspace(q[0], q[-1], 401)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].plot(q, measured_energy, ".", color="black", label="Noisy energies")
    axes[0].plot(fine_q, spline(fine_q), label="Interpolating spline")
    axes[0].plot(fine_q, fit(fine_q), label="Quadratic fit")
    axes[0].plot(fine_q, 0.5 * stiffness * fine_q**2, "k--", label="Exact")
    axes[0].set(xlabel="Displacement q (Å)", ylabel="Energy (eV)")
    axes[1].plot(q, raw_force, ".-", label="Raw finite difference")
    axes[1].plot(fine_q, -spline.derivative()(fine_q), label="Spline derivative")
    axes[1].plot(fine_q, -fit.deriv()(fine_q), label="Fit derivative")
    axes[1].plot(fine_q, -stiffness * fine_q, "k--", label="Exact")
    axes[1].set(xlabel="Displacement q (Å)", ylabel="Force (eV/Å)")
    for _ax in axes:
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.25)
    fig
    return


@app.cell
def _(derivative, fit, fitted_force, noise_eV, np, q, raw_force, spline_force, stiffness):
    print("Force RMSE on interior sample points (eV/Å):")
    for _name, _force in [("Raw FD", raw_force), ("Spline", spline_force), ("Quadratic fit", fitted_force)]:
        _error = (_force + stiffness * q)[1:-1]
        print(f"{_name:14s}: {np.sqrt(np.mean(_error**2)):.6g}")
    print("Predicted raw central-difference noise SD:", noise_eV / (np.sqrt(2) * (q[1] - q[0])), "eV/Å")
    probe = 0.1  # Å
    result = derivative(fit, probe, initial_step=0.05)
    print("\nAt q = 0.1 Å:")
    print("Exact model force        :", -stiffness * probe, "eV/Å")
    print("Analytical fit derivative:", -fit.deriv()(probe), "eV/Å")
    print("SciPy FD of the fit      :", -result.df, "eV/Å")
    print("SciPy success:", result.success, " estimated differentiation error:", result.error)
    print("Energy-fit residual RMS :", np.sqrt(np.mean((fit(q) - measured_energy)**2)), "eV")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Interpretation.** The spline retains every noisy energy, so its analytical
    derivative also fluctuates. The quadratic fit pools the data using the
    harmonic model. Set the noise to zero to check all three routes against
    $F=-kq$, then increase the sample count at fixed nonzero noise.
    `scipy.differentiate` can agree with the fit's analytical derivative while
    both differ from the true force. Its reported error measures the numerical
    differentiation of the fit, not the uncertainty in the fitted model.
    """)
    return


if __name__ == "__main__":
    app.run()

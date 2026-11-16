# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import root_scalar

    return mo, np, plt, root_scalar


@app.cell
def _(np):
    # Funnel of the UQ experiment (Edgeworth, Dalton and Parnell, 1984)
    RHO, G = 1.1e3, 9.81                 # density of pitch (kg/m^3), gravity (m/s^2)
    D, L, H = 0.94e-2, 2.9e-2, 7.5e-2    # stem diameter, stem length, pitch depth (m)
    V_DROP = 4.7e-5 / 6                  # volume of one drop (m^3): 6 drops by 1979
    YEAR = 3.156e7                       # s


    def drop_time(mu):
        """Hagen-Poiseuille model: years for one drop at viscosity mu (Pa s)."""
        Q = np.pi * RHO * G * D**4 * (1 + H / L) / (128 * mu)   # m^3/s
        return V_DROP / Q / YEAR

    return (drop_time,)


@app.cell(hide_code=True)
def _(mo):
    observed = mo.ui.dropdown(
        options={"drops 1-6, 1930-1979 (8.1 years)": 8.08,
                 "drop 8, 1988-2000 (12.3 years)": 12.3,
                 "drop 9, 2000-2014 (13.4 years)": 13.4},
        value="drops 1-6, 1930-1979 (8.1 years)", label="observed time per drop")
    return (observed,)


@app.cell
def _(drop_time, observed, root_scalar):
    t_observed = observed.value


    def residual(log_mu):
        """Model minus observed drop time; the unknown is log10 of the viscosity."""
        return drop_time(10**log_mu) - t_observed


    result = root_scalar(residual, bracket=[0.0, 14.0], method="brentq", xtol=1e-8)   # 1 to 1e14 Pa s
    mu_root = 10**result.root
    print(result.converged, result.function_calls, mu_root)
    return mu_root, residual, result, t_observed


@app.cell(hide_code=True)
def _(mo, mu_root, np, observed, plt, residual, result, t_observed):
    _mu = np.logspace(5, 11, 300)
    _fig, _ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    _ax.semilogx(_mu, [residual(np.log10(m)) for m in _mu], color="C0")
    _ax.axhline(0, color="0.5", lw=1)
    _ax.semilogx([mu_root], [0], "*", color="C3", ms=16, label="root")
    _ax.set(xlabel="viscosity μ (Pa·s)", ylabel="model − observed (years)", ylim=(-20, 40))
    _ax.grid(alpha=0.25)
    _ax.legend(fontsize=8)
    mo.vstack([observed,
               mo.md(f"Root: **μ = {mu_root:.3g} Pa·s**, about {mu_root / 1e-3:.1g} times water, "
                     f"from brentq with {result.function_calls} function calls, for a drop every {t_observed:g} years."),
               _fig])
    return


if __name__ == "__main__":
    app.run()

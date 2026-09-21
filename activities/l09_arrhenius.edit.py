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
    from scipy.optimize import curve_fit
    return curve_fit, mo, np, plt


@app.cell(hide_code=True)
def controls(mo):
    weighting = mo.ui.dropdown(["Equal absolute uncertainty", "Equal relative uncertainty"], value="Equal absolute uncertainty", label="Direct-fit residual weighting")
    weighting
    return (weighting,)


@app.cell
def data(np):
    # Synthetic rate data: K and s⁻¹. Fixed scatter makes comparisons repeatable.
    T = np.arange(500., 701., 25.)
    R = 0.008314462618  # kJ mol⁻¹ K⁻¹; Q below is in kJ/mol.
    k = 1e7 * np.exp(-80 / (R*T)) * np.array([1.22, 0.78, 1.12, 0.90, 1.25, 0.82, 1.08, 1.17, 0.88])
    return R, T, k


@app.cell
def linearized(R, T, k, np):
    slope, log_A = np.polyfit(1/T, np.log(k), deg=1)
    Q_linear = -R * slope
    A_linear = np.exp(log_A)
    k_linear = A_linear * np.exp(-Q_linear / (R*T))
    print(f"Log-space fit: A = {A_linear:.3g} s⁻¹, Q = {Q_linear:.3f} kJ/mol")
    return A_linear, Q_linear, k_linear, log_A


@app.cell
def nonlinear(Q_linear, R, T, curve_fit, k, log_A, np, weighting):
    def arrhenius(temperature, log_prefactor, Q):
        return np.exp(log_prefactor - Q/(R*temperature))

    # log(A) and Q in kJ/mol keep the parameter scales manageable.
    sigma = np.ones_like(k) if weighting.value == "Equal absolute uncertainty" else 0.15*k
    parameters, covariance = curve_fit(
        arrhenius, T, k, p0=(log_A, Q_linear), sigma=sigma,
    )
    A_direct = np.exp(parameters[0])
    Q_direct = parameters[1]
    k_direct = arrhenius(T, *parameters)
    print(f"Direct fit: A = {A_direct:.3g} s⁻¹, Q = {Q_direct:.3f} kJ/mol")
    return A_direct, Q_direct, arrhenius, covariance, k_direct, parameters, sigma


@app.cell(hide_code=True)
def compare(A_linear, Q_linear, R, T, arrhenius, k, k_direct, k_linear, mo, np, parameters, plt, sigma):
    arrhenius_figure, _axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    _grid = np.linspace(T.min(), T.max(), 300)
    _curves = [("Linearized", k_linear, A_linear*np.exp(-Q_linear/(R*_grid))), ("Direct", k_direct, arrhenius(_grid, *parameters))]
    _axes[0, 0].scatter(T, k, color="black", s=22, label="Synthetic data")
    _axes[0, 1].scatter(1000/T, np.log(k), color="black", s=22)
    _rows = []
    for _label, _predicted, _curve in _curves:
        _axes[0, 0].plot(_grid, _curve, label=_label)
        _axes[0, 1].plot(1000/_grid, np.log(_curve), label=_label)
        _axes[1, 0].plot(T, k-_predicted, "o-", label=_label)
        _axes[1, 1].plot(T, np.log(k)-np.log(_predicted), "o-", label=_label)
        _rows.append(f"| {_label} | {np.sum((k-_predicted)**2):.4g} | {np.sum((np.log(k)-np.log(_predicted))**2):.4g} | {np.sum(((k-_predicted)/sigma)**2):.4g} |")
    _axes[0, 0].set(xlabel="Temperature (K)", ylabel="Rate constant (s⁻¹)", title="Original data space")
    _axes[0, 1].set(xlabel="1000 / T (K⁻¹)", ylabel="ln[k / (1 s⁻¹)]", title="Arrhenius plot")
    _axes[1, 0].set(xlabel="Temperature (K)", ylabel="Rate residual (s⁻¹)")
    _axes[1, 1].set(xlabel="Temperature (K)", ylabel="Log-rate residual")
    for _ax in _axes[1]:
        _ax.axhline(0, color="0.4", lw=0.8)
    for _ax in _axes.flat:
        _ax.grid(alpha=0.2)
        _ax.legend(fontsize=8)
    mo.vstack([arrhenius_figure, mo.md("| Fit | Rate-space SSE (s⁻²) | Log-space SSE | Direct weighted objective |\n|---|---:|---:|---:|\n" + "\n".join(_rows))])
    return (arrhenius_figure,)


if __name__ == "__main__":
    app.run()

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
    from time import perf_counter
    from scipy.optimize import brentq
    from scipy.interpolate import CubicSpline, PchipInterpolator
    return CubicSpline, PchipInterpolator, brentq, mo, np, perf_counter, plt


@app.cell(hide_code=True)
def model(brentq, np):
    # Same dimensional vdW CO₂ model as L07: L/mol, K, bar, J/mol.
    a, b, R = 3.592, 0.04267, 0.08314462618
    critical_T = 8*a/(27*R*b)
    critical_P = a/(27*b*b)

    def pressure_eos(v, T):
        return R*T/(v-b) - a/v**2

    def stationary_volumes(T, P):
        roots = np.roots([P, -(P*b + R*T), a, -a*b])
        return np.sort([z.real for z in roots if abs(z.imag) < 1e-8 and z.real > b])

    def gibbs(v, T, P):
        return 100*(-R*T*np.log((v-b)/1.0) - a/v + P*v)

    def delta_g(T, P):
        volumes = stationary_volumes(T, P)
        if len(volumes) != 3:
            return np.nan  # Two distinct minima are needed for this comparison.
        return float(gibbs(volumes[0], T, P) - gibbs(volumes[-1], T, P))

    def saturation_pressure(T):
        if not 245 <= T <= critical_T - 0.05:
            return np.nan  # Tested subcritical range for this classroom calculation.
        roots = np.roots([R*T, -2*a, 4*a*b, -2*a*b*b])
        turning = sorted(z.real for z in roots if abs(z.imag) < 1e-8 and z.real > b)
        low = max(1e-5, pressure_eos(turning[0], T))
        high = pressure_eos(turning[1], T)
        margin = (high-low)*1e-7
        return brentq(lambda P: delta_g(T, P), low+margin, high-margin, xtol=1e-10)

    return critical_P, critical_T, delta_g, saturation_pressure, stationary_volumes


@app.cell(hide_code=True)
def reference(critical_T, np, saturation_pressure):
    reference_T = np.linspace(250, critical_T-0.05, 101)
    reference_P = np.array([saturation_pressure(t) for t in reference_T])
    return reference_P, reference_T


@app.cell(hide_code=True)
def controls(mo):
    sample_count = mo.ui.slider(4, 10, step=1, value=6, label="Calculated boundary points", show_value=True)
    query_temperature = mo.ui.slider(250, 310, step=1, value=280, label="Query temperature (K)", show_value=True)
    phase_method = mo.ui.dropdown(["Piecewise linear", "Natural cubic spline", "PCHIP"], value="Natural cubic spline", label="Interpolant")
    allow_extrapolation = mo.ui.checkbox(value=False, label="Extend cubic / PCHIP beyond the table")
    mo.vstack([mo.hstack([sample_count, query_temperature], wrap=True), mo.hstack([phase_method, allow_extrapolation], wrap=True)])
    return allow_extrapolation, phase_method, query_temperature, sample_count


@app.cell(hide_code=True)
def sample(np, perf_counter, sample_count, saturation_pressure):
    sample_T = np.linspace(250, 295, sample_count.value)
    _start = perf_counter()
    sample_P = np.array([saturation_pressure(t) for t in sample_T])
    table_seconds = perf_counter() - _start
    return sample_P, sample_T, table_seconds


@app.cell
def interpolant(CubicSpline, PchipInterpolator, allow_extrapolation, np, phase_method, sample_P, sample_T):
    if phase_method.value == "Piecewise linear":
        def estimate_pressure(T):
            return np.interp(T, sample_T, sample_P, left=np.nan, right=np.nan)
    elif phase_method.value == "Natural cubic spline":
        estimate_pressure = CubicSpline(sample_T, sample_P, bc_type="natural", extrapolate=allow_extrapolation.value)
    else:
        estimate_pressure = PchipInterpolator(sample_T, sample_P, extrapolate=allow_extrapolation.value)
    return (estimate_pressure,)


@app.cell(hide_code=True)
def check(delta_g, estimate_pressure, np, perf_counter, query_temperature, sample_P, sample_T, saturation_pressure):
    query_K = query_temperature.value
    estimated_P = float(estimate_pressure(query_K))
    direct_P = saturation_pressure(query_K)
    energy_mismatch = delta_g(query_K, estimated_P) if np.isfinite(estimated_P) else np.nan
    node_residual = float(np.max(np.abs(estimate_pressure(sample_T)-sample_P)))
    _check_T = np.linspace(250, 295, 1000)
    _start = perf_counter()
    estimate_pressure(_check_T)
    lookup_seconds = perf_counter() - _start
    return direct_P, energy_mismatch, estimated_P, lookup_seconds, node_residual, query_K


@app.cell(hide_code=True)
def plot(critical_P, critical_T, direct_P, energy_mismatch, estimate_pressure, estimated_P, lookup_seconds, mo, node_residual, np, plt, query_K, reference_P, reference_T, sample_P, sample_T, table_seconds):
    phase_figure, _axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    _axes[0].plot(reference_T, reference_P, "k--", label="Direct coexistence solves")
    _inside = np.linspace(sample_T[0], sample_T[-1], 400)
    _axes[0].plot(_inside, estimate_pressure(_inside), label="Interpolated boundary")
    _outside = np.linspace(sample_T[-1], 310, 160)
    _axes[0].plot(_outside, estimate_pressure(_outside), ":", color="tab:blue", label="Extension, if enabled")
    _axes[0].scatter(sample_T, sample_P, color="black", s=25, zorder=4, label="Points used")
    _axes[0].scatter(critical_T, critical_P, marker="*", s=85, color="tab:red", label="vdW critical point")
    if np.isfinite(estimated_P):
        _axes[0].scatter(query_K, estimated_P, marker="x", s=70, color="tab:orange", zorder=5)
    _axes[0].axvspan(295, 310, color="0.5", alpha=0.1)
    _axes[0].set(xlabel="Temperature (K)", ylabel="Pressure (bar)", title="CO₂ model: sparse coexistence data", xlim=(248, 311))
    _mask = reference_T <= 295
    _axes[1].plot(reference_T[_mask], estimate_pressure(reference_T[_mask])-reference_P[_mask])
    _axes[1].axhline(0, color="0.4", lw=0.8)
    _axes[1].set(xlabel="Temperature (K)", ylabel="Interpolated − direct pressure (bar)", title="Error on independent check points")
    _axes[0].legend(fontsize=7)
    for _ax in _axes:
        _ax.grid(alpha=0.2)
    _report = f"At **{query_K:g} K**: interpolated **{estimated_P:.5f} bar**; direct **{direct_P:.5f} bar**; ΔG at interpolated P **{energy_mismatch:+.3g} J/mol**."
    if query_K >= critical_T:
        _report += " **Above Tc: there is no liquid–gas coexistence pressure.**"
    elif query_K > sample_T[-1]:
        _report += " **Outside the table: this is extrapolation.**"
    _report += f"\n\nMaximum node residual: **{node_residual:.1e} bar**. Build {len(sample_T)} boundary points: **{1000*table_seconds:.2f} ms**; evaluate 1000 interpolated values: **{1000*lookup_seconds:.3f} ms** (device-dependent; reference-plot work excluded)."
    mo.vstack([phase_figure, mo.md(_report)])
    return (phase_figure,)


if __name__ == "__main__":
    app.run()

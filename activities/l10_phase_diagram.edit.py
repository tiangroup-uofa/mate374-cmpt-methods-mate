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
    from scipy.optimize import root_scalar, minimize_scalar
    from scipy.interpolate import PchipInterpolator, CubicSpline

    return CubicSpline, PchipInterpolator, minimize_scalar, mo, np, plt, root_scalar


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## L10 · A phase diagram from one function

    **Starter notebook.** The two editable tasks are `solve_pressure` and
    `boundary`. The free-energy, interval-finding, minimization, and plotting
    functions are supplied and collapsed. Edit existing cells to avoid duplicate names.

    Start with CO₂, then select another real substance. The same solver accepts its
    `a` and `b`; the trial temperatures adjust to that model's critical temperature.
    The comparison table also tests six substances against NIST vapour-pressure
    correlations based on experimental data.

    Work in pairs: one person types, the other checks inputs, outputs, and units.
    Save before each major edit. We stop at 35 minutes for Wooclap.
    """)
    return


@app.cell(hide_code=True)
def supplied_gas_data():
    # Offline copy of data/L10-gases.json. Sources and validity ranges are on the lecture page.
    # a: L² bar/mol²; b: L/mol; Antoine pressure: bar, temperature: K.
    gases = {
        "CO₂": {"a": 3.592, "b": 0.04267, "reference": None},
        "Argon": {"a": 1.355, "b": 0.03201, "reference":
                  {"A": 3.29555, "B": 215.24, "C": -22.233, "T_min": 83.78, "T_max": 150.72}},
        "Nitrogen": {"a": 1.370, "b": 0.0387, "reference":
                     {"A": 3.7362, "B": 264.651, "C": -6.788, "T_min": 63.14, "T_max": 126.0}},
        "Oxygen": {"a": 1.382, "b": 0.03186, "reference":
                   {"A": 3.9523, "B": 340.024, "C": -4.144, "T_min": 54.36, "T_max": 154.33}},
        "Methane": {"a": 2.283, "b": 0.04278, "reference":
                    {"A": 3.9895, "B": 443.028, "C": -0.49, "T_min": 90.99, "T_max": 189.99}},
        "Ethane": {"a": 5.562, "b": 0.0638, "reference": None},
        "Ammonia": {"a": 4.225, "b": 0.0371, "reference":
                    {"A": 4.86886, "B": 1113.928, "C": -10.409, "T_min": 239.6, "T_max": 371.5}},
        "Water": {"a": 5.536, "b": 0.03049, "reference":
                  {"A": 3.55959, "B": 643.748, "C": -198.043, "T_min": 379.0, "T_max": 573.0}},
    }

    def reference_pressure(gas, T):
        ref = gas["reference"]
        if ref is None:
            return None
        if not ref["T_min"] <= T <= ref["T_max"]:
            raise ValueError("Temperature outside this NIST Antoine correlation's range.")
        return 10**(ref["A"]-ref["B"]/(T+ref["C"]))

    return gases, reference_pressure


@app.cell(hide_code=True)
def gas_selector(gases, mo):
    gas_choice = mo.ui.dropdown(list(gases), value="CO₂", label="Substance")
    gas_choice
    return (gas_choice,)


@app.cell(hide_code=False)
def parameters(R, gas_choice, gases):
    gas_name = gas_choice.value
    a = gases[gas_name]["a"]  # L² bar / mol²
    b = gases[gas_name]["b"]  # L / mol
    Tc = 8*a/(27*R*b)        # Model critical temperature, K
    print(gas_name, "a =", a, "b =", b, "model Tc =", Tc, "K")
    return Tc, a, b, gas_name


@app.cell(hide_code=True)
def supplied_model(np):
    R = 0.08314462618  # L bar / (mol K)

    def free_energy(v, T, P, a, b):
        # v in L/mol, T in K, P in bar; output in J/mol.
        # v_ref = 1 L/mol, and 1 L bar = 100 J. C(T) cancels in differences.
        return 100 * (-R*T*np.log((v-b)/1.0) - a/v + P*v)

    def pressure_eos(v, T, a, b):
        return R*T/(v-b) - a/v**2

    return R, free_energy, pressure_eos


@app.cell(hide_code=True)
def supplied_intervals(R, np, pressure_eos, root_scalar):
    def turning_volumes(T, a, b):
        if not np.all(np.isfinite([T, a, b])) or min(T, a, b) <= 0:
            raise ValueError("T, a, and b must be finite and positive.")
        Tc = 8*a/(27*R*b)
        if T >= Tc:
            raise ValueError("No distinct liquid–gas coexistence at or above the model Tc.")
        if not 0.82 <= T/Tc <= 0.985:
            raise ValueError("This classroom scaffold supports 0.82 <= T/Tc <= 0.985.")

        # The two EOS turning points separate the liquid and gas wells.
        # Solve in the dimensionless volume w = v/b for numerical scaling.
        def slope(w):
            return -R*T*b/(a*(w-1)**2) + 2/w**3

        lower = root_scalar(slope, bracket=(1+1e-8, 3), method="brentq",
                            xtol=1e-12, maxiter=100)
        upper = root_scalar(slope, bracket=(3, 2*a/(R*T*b)), method="brentq",
                            xtol=1e-12, maxiter=100)
        if not lower.converged or not upper.converged:
            raise RuntimeError("The supplied turning-point calculation did not converge.")
        return b*lower.root, b*upper.root

    def find_pressure_bracket(T, a, b):
        v1, v2 = turning_volumes(T, a, b)
        Pc = a/(27*b*b)
        low = max(pressure_eos(v1, T, a, b), 1e-6*Pc)
        high = pressure_eos(v2, T, a, b)
        margin = 1e-4*(high-low)  # Keep two distinct wells at the trial endpoints.
        return low+margin, high-margin

    return find_pressure_bracket, turning_volumes


@app.cell(hide_code=True)
def supplied_minima(R, minimize_scalar, np, pressure_eos, turning_volumes):
    def find_phase_volumes(g_func, T, P, a, b):
        if not np.isfinite(P) or P <= 0:
            raise ValueError("Use a finite, positive pressure.")
        v1, v2 = turning_volumes(T, a, b)
        if not pressure_eos(v1, T, a, b) < P < pressure_eos(v2, T, a, b):
            raise ValueError("This pressure does not have two distinct free-energy wells.")
        intervals = ((b*(1+1e-8), v1), (v2, 2*R*T/P + 2*b))
        volumes = []
        for left, right in intervals:
            # Bounded Brent minimization uses golden-section fallback steps.
            # Log(v/b - 1) resolves both liquid and dilute-gas volumes.
            answer = minimize_scalar(
                lambda q: g_func(b*(1+np.exp(q))),
                bounds=(np.log(left/b-1), np.log(right/b-1)), method="bounded",
                options={"xatol": 1e-11, "maxiter": 150},
            )
            if not answer.success:
                raise RuntimeError("A supplied free-energy minimization did not converge.")
            volumes.append(b*(1+np.exp(answer.x)))
        return tuple(volumes)

    return (find_phase_volumes,)


@app.cell(hide_code=True)
def supplied_residual(find_phase_volumes, free_energy):
    def free_energy_difference_P(P, T, a, b):
        def g_func(v):
            return free_energy(v, T, P, a, b)

        v_liq, v_gas = find_phase_volumes(g_func, T, P, a, b)
        return g_func(v_liq) - g_func(v_gas)

    return (free_energy_difference_P,)


@app.cell(hide_code=True)
def task_one(mo):
    mo.md(r"""
    ### 1. Complete the outer pressure solve

    `find_pressure_bracket(T, a, b)` supplies a valid pressure interval in bar.
    `residual(P)` supplies the liquid-minus-gas free energy in J/mol.
    Use `root_scalar` to make this residual zero and return the pressure in bar.
    Both functions already perform the intermediate calculations.
    """)
    return


@app.cell(hide_code=False)
def pressure_solver(find_pressure_bracket, free_energy_difference_P, root_scalar):
    def solve_pressure(T, a, b):
        pressure_bracket = find_pressure_bracket(T, a, b)

        def residual(P):
            return free_energy_difference_P(P, T, a, b)

        # YOUR CODE: solve residual(P) = 0 inside pressure_bracket.
        # Check result.converged, then return result.root (pressure in bar).
        return None

    return (solve_pressure,)


@app.cell(hide_code=False)
def three_trials(Tc, a, b, gas_name, np, solve_pressure):
    trial_T = ([250.0, 270.0, 290.0] if gas_name == "CO₂"
               else np.round(Tc * np.array([0.84, 0.90, 0.96]), 1))  # K
    trial_P = [solve_pressure(T, a, b) for T in trial_T]
    print("Coexistence pressures (bar):", trial_P)
    return trial_P, trial_T


@app.cell(hide_code=True)
def supplied_trial_checks(a, b, find_phase_volumes, free_energy, free_energy_difference_P, mo, np, pressure_eos, trial_P, trial_T):
    mo.stop(any(P is None for P in trial_P), mo.md(
        "**Complete `solve_pressure` above.** It currently returns `None`. "
        "The downstream calculation will start when it returns a pressure."
    ))

    def check_trials():
        lines = ["| T (K) | P (bar) | ΔG (J/mol) | Largest EOS residual (bar) |",
                 "|---:|---:|---:|---:|"]
        checks = []
        for T, P in zip(trial_T, trial_P):
            if not np.isfinite(P):
                raise ValueError("solve_pressure must return one finite pressure in bar.")
            g_func = lambda v: free_energy(v, T, P, a, b)
            volumes = np.array(find_phase_volumes(g_func, T, P, a, b))
            gap = free_energy_difference_P(P, T, a, b)
            eos_error = np.max(np.abs(pressure_eos(volumes, T, a, b)-P))
            checks.append(abs(gap) < 1e-4 and eos_error < 1e-3)
            lines.append(f"| {T:g} | {P:.6f} | {gap:+.2e} | {eos_error:.2e} |")
        return all(checks), "\n".join(lines)

    checks_passed, trial_report = check_trials()
    mo.vstack([mo.md(trial_report), mo.callout(
        "Trial checks passed." if checks_passed else
        "Check your pressure solve: require |ΔG| < 1e-4 J/mol and EOS residual < 1e-3 bar.",
        kind="success" if checks_passed else "warn",
    )])
    return (checks_passed,)


@app.cell(hide_code=True)
def task_two(mo):
    mo.md(r"""
    ### 2. From a few pressures to a curve

    The next cell reuses your function at five temperatures. Complete `boundary`
    with a `PchipInterpolator` using these temperatures and pressures, and disable
    extrapolation. The supplied plotting and fitting cells then run automatically.
    """)
    return


@app.cell(hide_code=False)
def boundary_points(a, b, checks_passed, mo, np, solve_pressure, trial_T):
    mo.stop(not checks_passed)  # Continue after the three trial checks pass.
    sample_T = np.linspace(trial_T[0], trial_T[-1], 5)
    sample_P = np.array([solve_pressure(T, a, b) for T in sample_T])
    print(np.column_stack([sample_T, sample_P]))
    return sample_P, sample_T


@app.cell(hide_code=False)
def interpolation(PchipInterpolator, sample_P, sample_T):
    # YOUR CODE: interpolate sample_P against sample_T; disable extrapolation.
    boundary = None
    return (boundary,)


@app.cell(hide_code=True)
def supplied_fit(np, sample_P, sample_T):
    # Fit ln[P/(1 bar)] = C - D/T. All T are in kelvin.
    slope, C = np.polyfit(1/sample_T, np.log(sample_P), deg=1)
    D = -slope  # K

    def fitted_pressure(T):
        return np.exp(C-D/np.asarray(T))  # bar

    return C, D, fitted_pressure


@app.cell(hide_code=True)
def independent_check(a, b, boundary, fitted_pressure, free_energy_difference_P, mo, sample_T, solve_pressure):
    mo.stop(boundary is None, mo.md(
        "**Complete `boundary` above.** Use the two sample arrays in `PchipInterpolator`."
    ))
    check_T = sample_T[0] + 0.625*(sample_T[-1]-sample_T[0])  # Between sample nodes; 275 K for CO₂.
    direct_P = solve_pressure(check_T, a, b)
    interpolated_P = float(boundary(check_T))
    fitted_P = float(fitted_pressure(check_T))
    interpolated_gap = free_energy_difference_P(interpolated_P, check_T, a, b)
    fitted_gap = free_energy_difference_P(fitted_P, check_T, a, b)
    mo.md(f"""
    ### Independent check at {check_T:g} K

    | Calculation | Pressure (bar) | Difference from direct solve (bar) | ΔG (J/mol) |
    |---|---:|---:|---:|
    | Direct solve | {direct_P:.6f} | 0 | {free_energy_difference_P(direct_P, check_T, a, b):+.3g} |
    | Interpolation | {interpolated_P:.6f} | {interpolated_P-direct_P:+.6f} | {interpolated_gap:+.3g} |
    | Fitted relation | {fitted_P:.6f} | {fitted_P-direct_P:+.6f} | {fitted_gap:+.3g} |

    This temperature was not used to construct either curve. Agreement here checks
    the approximation to the **vdW calculation**. The multi-gas comparison below
    separately checks predictions against correlations of experimental data.
    """)
    return check_T, direct_P, fitted_P, fitted_gap, interpolated_P, interpolated_gap


@app.cell(hide_code=True)
def supplied_plot(C, D, boundary, check_T, direct_P, fitted_pressure, gas_name, mo, np, plt, sample_P, sample_T):
    def plot_phase_diagram():
        grid = np.linspace(sample_T[0], sample_T[-1], 300)
        curve = boundary(grid)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_low, plot_high = 0.5*sample_P[0], 1.3*sample_P[-1]
        ax.fill_between(grid, plot_low, curve, color="tab:blue", alpha=0.10)
        ax.fill_between(grid, curve, plot_high, color="tab:orange", alpha=0.12)
        ax.plot(grid, curve, label="Interpolated boundary")
        ax.plot(grid, fitted_pressure(grid), "--", label="Fit: ln(P / bar) = C − D/T")
        ax.scatter(sample_T, sample_P, c="black", s=30, zorder=4, label="Five coexistence solves")
        ax.scatter([check_T], [direct_P], marker="*", c="crimson", s=130,
                   zorder=5, label=f"Independent solve at {check_T:g} K")
        ax.text(0.7, 0.87, "Liquid favoured", color="saddlebrown", ha="center", transform=ax.transAxes)
        ax.text(0.7, 0.15, "Gas favoured", color="navy", ha="center", transform=ax.transAxes)
        ax.set(xlabel="Temperature (K)", ylabel="Pressure (bar)",
               xlim=(sample_T[0], sample_T[-1]), ylim=(plot_low, plot_high),
               title=f"Liquid–gas boundary: vdW model for {gas_name}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    phase_figure = plot_phase_diagram()
    mo.vstack([phase_figure, mo.md(
        f"Supplied fit: **C = {C:.5f}**, **D = {D:.2f} K**. "
        "The approximate Clausius–Clapeyron interpretation is ΔH ≈ R D. "
        "Compare the fresh check point before choosing how to represent the curve."
    )])
    return (phase_figure,)


@app.cell(hide_code=True)
def supplied_real_gas_comparison(R, checks_passed, free_energy_difference_P, gases, mo, reference_pressure, solve_pressure):
    mo.stop(not checks_passed)

    def compare_gases():
        rows = []
        for name, gas in gases.items():
            T = 0.85 * 8*gas["a"]/(27*R*gas["b"])
            P = solve_pressure(T, gas["a"], gas["b"])
            reference = reference_pressure(gas, T)
            rows.append(dict(gas=name, T=T, P=P,
                             gap=free_energy_difference_P(P, T, gas["a"], gas["b"]),
                             reference=reference,
                             percent=None if reference is None else 100*(P/reference-1)))
        return rows

    gas_comparison = compare_gases()
    comparison_lines = ["### One solver, eight substances",
                        "At **0.85 times each model's critical temperature**:",
                        "| Substance | T (K) | vdW P (bar) | ΔG (J/mol) | NIST correlation P (bar) | Difference (%) |",
                        "|---|---:|---:|---:|---:|---:|"]
    for row in gas_comparison:
        reference_text = "—" if row["reference"] is None else f"{row['reference']:.4f}"
        percent_text = "—" if row["percent"] is None else f"{row['percent']:+.1f}"
        comparison_lines.append(f"| {row['gas']} | {row['T']:.2f} | {row['P']:.4f} | {row['gap']:+.1e} | {reference_text} | {percent_text} |")
    comparison_lines.append(
        "\nA small ΔG checks the **numerical solution**. The pressure discrepancy tests "
        "the **vdW prediction** against a separate empirical correlation. These NIST "
        "correlations summarize experimental data and have finite validity ranges. "
        "The reference column is intentionally blank for CO₂ and ethane: the selected "
        "WebBook Antoine tables do not cover this liquid–gas comparison temperature."
    )
    mo.md("\n\n".join(comparison_lines[:2]) + "\n\n" + "\n".join(comparison_lines[2:]))
    return (gas_comparison,)


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import io
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit, minimize_scalar, root_scalar
    from scipy.interpolate import PchipInterpolator

    return PchipInterpolator, curve_fit, io, minimize_scalar, mo, np, plt, root_scalar


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## L10 · From CO₂ gas measurements to a liquid–gas boundary

    **Starter notebook.** The three editable tasks are the fit of `a_fit, b_fit`,
    `solve_pressure`, and the `boundary` interpolation. The data, free-energy
    functions, plots, and checks are supplied and collapsed. Edit the existing
    cells rather than defining the same names in a new cell.

    **Engineering question.** CO₂ is held in a line or vessel at a temperature between
    250 and 295 K. Above which pressure does it condense to liquid? We answer with
    the van der Waals (vdW) model, whose two parameters come only from gas-phase
    measurements at 300–330 K.

    Work in pairs: one person types, the other checks inputs, outputs, and units.
    Save before each major edit. We stop at 35 minutes for Wooclap.
    """)
    return


@app.cell(hide_code=True)
def supplied_data(io, np):
    # Offline copy of data/L10-co2-pvt.csv: CO₂ isochoric (P, ρ, T) measurements
    # of Ely, Haynes and Bain (1989). Columns: T (K), P (MPa), ρ (mol/L), subset.
    csv_text = """T_K,P_MPa,rho_mol_L,subset
    300.0,2.1936,0.99431,fit_batch
    300.0,5.07784,2.99455,fit_batch
    305.0,7.33917,6.86619,fit_batch
    310.0,2.28675,0.99322,fit_batch
    310.0,5.43213,2.99001,fit_batch
    310.0,7.12164,4.98887,additional_conditions
    310.0,7.86433,6.85824,fit_batch
    310.0,8.06822,7.83874,additional_conditions
    310.0,8.20823,8.82938,additional_conditions
    310.0,8.31622,9.87864,fit_batch
    310.0,8.44698,11.24963,additional_conditions
    310.0,8.62861,12.59945,additional_conditions
    310.0,9.02537,14.05004,additional_conditions
    310.0,10.37431,15.95375,additional_conditions
    320.0,2.37898,0.99211,fit_batch
    320.0,5.77614,2.98536,fit_batch
    320.0,7.79098,4.97857,additional_conditions
    320.0,8.87599,6.84063,fit_batch
    320.0,9.26616,7.8168,additional_conditions
    320.0,9.59111,8.80289,additional_conditions
    320.0,9.89966,9.84698,fit_batch
    320.0,10.3143,11.21009,additional_conditions
    320.0,10.82856,12.54955,additional_conditions
    320.0,11.67866,13.98434,additional_conditions
    320.0,13.81104,15.87138,additional_conditions
    330.0,2.47023,0.991,fit_batch
    330.0,6.11169,2.97716,fit_batch
    330.0,8.43934,4.96743,additional_conditions
    330.0,9.85303,6.81963,fit_batch
    330.0,10.42414,7.78929,additional_conditions
    330.0,10.94096,8.76818,additional_conditions
    330.0,11.46213,9.80457,fit_batch
    330.0,12.1827,11.15849,additional_conditions
    330.0,13.0513,12.49189,additional_conditions
    330.0,14.37347,13.92777,additional_conditions
    330.0,17.32428,15.82941,additional_conditions"""
    table = np.genfromtxt(io.StringIO(csv_text.replace("    ", "")), delimiter=",",
                          names=True, dtype=None, encoding="utf-8")
    T_data = table["T_K"].astype(float)            # K
    P_data = 10*table["P_MPa"].astype(float)       # bar
    rho_data = table["rho_mol_L"].astype(float)    # mol/L
    v_data = 1/rho_data                            # L/mol
    fit_set = table["subset"] == "fit_batch"       # 15 states, all below 10 mol/L
    T_fit, v_fit, P_fit = T_data[fit_set], v_data[fit_set], P_data[fit_set]
    print(f"{fit_set.sum()} fitting states; largest density {rho_data[fit_set].max():.2f} mol/L")
    return P_data, P_fit, T_data, T_fit, fit_set, rho_data, v_data, v_fit


@app.cell(hide_code=True)
def supplied_model(np):
    R = 0.08314462618  # L bar / (mol K)

    def pressure_eos(v, T, a, b):
        # vdW pressure in bar; v in L/mol, T in K.
        return R*T/(v-b) - a/v**2

    def vdw_pressure(state, a, b):
        # curve_fit form: the known state (T, v) comes first, then the unknown a, b.
        T, v = state
        return pressure_eos(v, T, a, b)

    def free_energy(v, T, P, a, b):
        # v in L/mol, T in K, P in bar; output in J/mol.
        # v_ref = 1 L/mol, and 1 L bar = 100 J. C(T) cancels in differences.
        return 100 * (-R*T*np.log((v-b)/1.0) - a/v + P*v)

    a_book, b_book = 3.592, 0.04267  # Tabulated CO₂ parameters used in L05–L09
    return R, a_book, b_book, free_energy, pressure_eos, vdw_pressure


@app.cell(hide_code=True)
def supplied_data_plot(P_data, T_data, fit_set, mo, np, plt, rho_data):
    def plot_data():
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for T in np.unique(T_data):
            on = T_data == T
            line, = ax.plot(rho_data[on], P_data[on], "-", lw=0.8, alpha=0.5)
            ax.plot(rho_data[on & fit_set], P_data[on & fit_set], "o",
                    color=line.get_color(), label=f"{T:g} K")
            ax.plot(rho_data[on & ~fit_set], P_data[on & ~fit_set], "o",
                    mfc="none", color=line.get_color())
        ax.axvline(10, color="gray", ls=":")
        ax.text(10.2, 25, "10 mol/L", color="gray")
        ax.set(xlabel="Density ρ (mol/L)", ylabel="Pressure P (bar)",
               title="CO₂ measurements: filled = fitting states, open = held back")
        ax.legend(fontsize=8, title="Isotherm")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    data_figure = plot_data()
    mo.vstack([mo.md("### The measurements"), data_figure])
    return (data_figure,)


@app.cell(hide_code=True)
def task_one(mo):
    mo.md(r"""
    ### 1. Fit the vdW parameters to the gas measurements

    `vdw_pressure((T, v), a, b)` returns the vdW pressure in bar. Use `curve_fit`
    with the fitting arrays `T_fit`, `v_fit`, `P_fit` and the starting guess
    `p0=(3.6, 0.043)` to find `a_fit` (L² bar/mol²) and `b_fit` (L/mol).
    """)
    return


@app.cell(hide_code=False)
def fit(P_fit, T_fit, curve_fit, v_fit, vdw_pressure):
    # YOUR CODE: fit vdw_pressure to the states (T_fit, v_fit) and pressures P_fit,
    # starting from p0=(3.6, 0.043). Unpack the two fitted parameters.
    a_fit, b_fit = None, None
    return a_fit, b_fit


@app.cell(hide_code=True)
def supplied_fit_check(P_data, P_fit, R, T_data, T_fit, a_book, a_fit, b_book, b_fit, curve_fit, fit_set, mo, np, pressure_eos, rho_data, v_data, v_fit, vdw_pressure):
    mo.stop(a_fit is None, mo.md(
        "**Complete the fit above.** `a_fit, b_fit` are still `None`."
    ))

    # Fitting every state, including the dense ones, shows why the fit set stops at 10 mol/L.
    a_all, b_all = curve_fit(vdw_pressure, (T_data, v_data), P_data, p0=(3.6, 0.043))[0]
    held_dilute = ~fit_set & (rho_data < 10)
    held_dense = rho_data > 10

    def rms(mask, a, b):
        return np.sqrt(np.mean((pressure_eos(v_data[mask], T_data[mask], a, b) - P_data[mask])**2))

    def fit_row(label, a, b):
        Tc, Pc = 8*a/(27*R*b), a/(27*b*b)
        return (f"| {label} | {a:.5f} | {b:.6f} | {rms(fit_set, a, b):.2f} | "
                f"{rms(held_dilute, a, b):.2f} | {rms(held_dense, a, b):.2f} | {Tc:.1f} | {Pc:.1f} |")

    mo.md("\n".join([
        "### How well does each parameter set describe the measurements?",
        "",
        "Root-mean-square pressure error in bar. The last two columns are the model critical point; "
        "measured CO₂ has T<sub>c</sub> = 304.1 K and P<sub>c</sub> = 73.8 bar.",
        "",
        f"| Parameters | a (L² bar/mol²) | b (L/mol) | Fitting states ({fit_set.sum()}) | "
        f"Held back, ρ < 10 ({held_dilute.sum()}) | Held back, ρ > 10 ({held_dense.sum()}) | "
        "T<sub>c</sub> (K) | P<sub>c</sub> (bar) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        fit_row("Fitted, ρ < 10 mol/L", a_fit, b_fit),
        fit_row("Tabulated (L05–L09)", a_book, b_book),
        fit_row("Fitted to all 36 states", a_all, b_all),
    ]))
    return a_all, b_all


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
def well_controls(a_fit, b_fit, find_pressure_bracket, mo):
    mo.stop(a_fit is None)
    well_T = 280.0  # K
    well_low, well_high = find_pressure_bracket(well_T, a_fit, b_fit)
    well_P = mo.ui.slider(float(round(well_low + 1)), float(round(well_high - 1)), step=0.1,
                          value=50.0, label="Trial pressure at 280 K (bar)",
                          show_value=True, debounce=True)
    mo.vstack([mo.md(r"""
    ### 2. Minimization: two free-energy wells at one trial pressure

    At fixed $T$ and $P$, each phase sits at the bottom of its own free-energy well.
    Move the pressure and watch which well is lower. The coexistence pressure is
    the one that makes the two minima equal.
    """), well_P])
    return well_P, well_T


@app.cell(hide_code=True)
def supplied_well_plot(a_fit, b_fit, find_phase_volumes, free_energy, mo, np, plt, well_P, well_T):
    def plot_wells(P):
        g_func = lambda v: free_energy(v, well_T, P, a_fit, b_fit)
        v_liq, v_gas = find_phase_volumes(g_func, well_T, P, a_fit, b_fit)
        g_liq, g_gas = g_func(v_liq), g_func(v_gas)
        v = np.geomspace(1.2*b_fit, 2*v_gas, 600)
        g = g_func(v)
        barrier = g[(v > v_liq) & (v < v_gas)].max()
        low = min(g_liq, g_gas)
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.plot(v, g - low, color="black", lw=1.2)
        ax.plot([v_liq], [g_liq - low], "o", color="tab:orange", ms=8, label=f"Liquid well: v = {v_liq:.4f} L/mol")
        ax.plot([v_gas], [g_gas - low], "o", color="tab:blue", ms=8, label=f"Gas well: v = {v_gas:.4f} L/mol")
        ax.set(xscale="log", xlabel="Molar volume v (L/mol, log scale)",
               ylabel="G − lower minimum (J/mol)", ylim=(-0.15*(barrier-low), 1.4*(barrier-low)),
               title=f"T = {well_T:g} K, P = {P:.1f} bar, fitted a and b")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        preferred = "liquid" if g_liq < g_gas else "gas"
        return fig, g_liq - g_gas, preferred

    well_figure, well_gap, well_phase = plot_wells(well_P.value)
    mo.vstack([well_figure, mo.md(
        f"ΔG = G<sub>liq</sub> − G<sub>gas</sub> = **{well_gap:+.1f} J/mol**, so the "
        f"**{well_phase}** is favoured at this pressure."
    )])
    return


@app.cell(hide_code=True)
def task_two(mo):
    mo.md(r"""
    ### 3. Root finding: the pressure that makes ΔG = 0

    `find_pressure_bracket(T, a, b)` supplies a valid pressure interval in bar.
    `residual(P)` returns the liquid-minus-gas free energy in J/mol, running both
    minimizations each time it is called. Use `root_scalar` to make this residual
    zero and return the pressure in bar.
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
def three_trials(a_fit, b_fit, mo, solve_pressure):
    mo.stop(a_fit is None)
    trial_T = [250.0, 270.0, 290.0]  # K
    trial_P = [solve_pressure(T, a_fit, b_fit) for T in trial_T]
    print("Coexistence pressures (bar):", trial_P)
    return trial_P, trial_T


@app.cell(hide_code=True)
def supplied_trial_checks(a_fit, b_fit, find_phase_volumes, free_energy, free_energy_difference_P, mo, np, pressure_eos, trial_P, trial_T):
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
            g_func = lambda v: free_energy(v, T, P, a_fit, b_fit)
            volumes = np.array(find_phase_volumes(g_func, T, P, a_fit, b_fit))
            gap = free_energy_difference_P(P, T, a_fit, b_fit)
            eos_error = np.max(np.abs(pressure_eos(volumes, T, a_fit, b_fit)-P))
            checks.append(abs(gap) < 1e-4 and eos_error < 1e-3)
            lines.append(f"| {T:g} | {P:.4f} | {gap:+.2e} | {eos_error:.2e} |")
        return all(checks), "\n".join(lines)

    checks_passed, trial_report = check_trials()
    mo.vstack([mo.md(trial_report), mo.callout(
        "Trial checks passed." if checks_passed else
        "Check your pressure solve: require |ΔG| < 1e-4 J/mol and EOS residual < 1e-3 bar.",
        kind="success" if checks_passed else "warn",
    )])
    return (checks_passed,)


@app.cell(hide_code=True)
def task_three(mo):
    mo.md(r"""
    ### 4. From five pressures to a phase boundary

    Call `solve_pressure` at each temperature in `sample_T` using the fitted
    parameters, then build `boundary` with `PchipInterpolator`, with extrapolation
    disabled. The supplied cells then compare the result with measured CO₂.
    """)
    return


@app.cell(hide_code=False)
def boundary_points(PchipInterpolator, a_fit, b_fit, checks_passed, mo, np, solve_pressure):
    mo.stop(not checks_passed)  # Continue after the three trial checks pass.
    sample_T = np.linspace(250.0, 295.0, 5)  # K
    # YOUR CODE: call solve_pressure(T, a_fit, b_fit) at each sample temperature,
    # then interpolate sample_P against sample_T with extrapolation disabled.
    sample_P = None
    boundary = None
    return boundary, sample_P, sample_T


@app.cell(hide_code=True)
def supplied_reference(np):
    # Measured CO₂ vapour pressure: Span and Wagner (1996) ancillary equation, Eq. 3.13.
    Tc_CO2, Pc_CO2 = 304.1282, 73.773  # K, bar

    def measured_saturation_pressure(T):
        tau = 1 - np.asarray(T)/Tc_CO2
        series = (-7.0602087*tau + 1.9391218*tau**1.5
                  - 1.6463597*tau**2 - 3.2995634*tau**4)
        return Pc_CO2*np.exp(Tc_CO2/np.asarray(T)*series)  # bar

    return Pc_CO2, Tc_CO2, measured_saturation_pressure


@app.cell(hide_code=True)
def supplied_answer(R, a_book, a_fit, b_book, b_fit, boundary, free_energy_difference_P, measured_saturation_pressure, mo, np, sample_T, solve_pressure):
    mo.stop(boundary is None, mo.md(
        "**Complete `boundary` above.** Use the two sample arrays in `PchipInterpolator`."
    ))
    check_T = 275.0  # K, not one of the five sample temperatures
    direct_P = solve_pressure(check_T, a_fit, b_fit)
    interpolated_P = float(boundary(check_T))
    interpolated_gap = free_energy_difference_P(interpolated_P, check_T, a_fit, b_fit)

    answer_T = np.array([250.0, 260.0, 270.0, 280.0, 290.0])
    answer_fit = np.array([solve_pressure(T, a_fit, b_fit) for T in answer_T])
    answer_book = np.array([solve_pressure(T, a_book, b_book) for T in answer_T])
    answer_measured = measured_saturation_pressure(answer_T)
    rows = [f"| {T:g} | {pf:.2f} | {pb:.2f} | {pm:.2f} | {100*(pf/pm-1):+.0f} |"
            for T, pf, pb, pm in zip(answer_T, answer_fit, answer_book, answer_measured)]
    mo.md("\n".join([
        f"### Interpolation check at {check_T:g} K",
        "",
        f"Direct solve **{direct_P:.4f} bar**, interpolated **{interpolated_P:.4f} bar**, "
        f"difference **{interpolated_P-direct_P:+.4f} bar**. At the interpolated pressure, "
        f"ΔG = {interpolated_gap:+.2f} J/mol.",
        "",
        "### Answer to the engineering question: condensation pressure",
        "",
        "| T (K) | vdW, fitted (bar) | vdW, tabulated (bar) | Measured CO₂ (bar) | Fitted vs measured (%) |",
        "|---:|---:|---:|---:|---:|",
        *rows,
    ]))
    return answer_T, answer_book, answer_fit, answer_measured, check_T, direct_P, interpolated_P


@app.cell(hide_code=True)
def supplied_phase_plot(P_data, Pc_CO2, R, T_data, Tc_CO2, a_book, a_fit, b_book, b_fit, boundary, check_T, direct_P, fit_set, measured_saturation_pressure, mo, np, plt, sample_P, sample_T, solve_pressure):
    def plot_phase_diagram():
        grid = np.linspace(sample_T[0], sample_T[-1], 300)
        book_T = np.linspace(250.0, 295.0, 10)
        book_P = [solve_pressure(T, a_book, b_book) for T in book_T]
        measured_T = np.linspace(245.0, Tc_CO2, 300)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(measured_T, measured_saturation_pressure(measured_T), color="black",
                lw=2, label="Measured CO₂ (Span–Wagner)")
        ax.plot(grid, boundary(grid), color="tab:red", lw=2, label="vdW, fitted a, b")
        ax.plot(book_T, book_P, color="tab:gray", ls="--", label="vdW, tabulated a, b")
        ax.scatter(sample_T, sample_P, c="tab:red", s=30, zorder=4, label="Five coexistence solves")
        ax.scatter([check_T], [direct_P], marker="*", c="gold", edgecolors="black",
                   s=160, zorder=5, label=f"Fresh solve at {check_T:g} K")
        for a, b, colour in [(a_fit, b_fit, "tab:red"), (a_book, b_book, "tab:gray")]:
            ax.plot([8*a/(27*R*b)], [a/(27*b*b)], "s", mfc="none", color=colour, ms=8)
        ax.plot([Tc_CO2], [Pc_CO2], "s", color="black", ms=7, label="Critical points")
        ax.scatter(T_data[fit_set], P_data[fit_set], marker="x", color="tab:blue",
                   label="States used to fit a, b")
        ax.text(262, 62, "Liquid", color="saddlebrown", fontsize=12)
        ax.text(282, 28, "Gas", color="navy", fontsize=12)
        ax.set(xlabel="Temperature (K)", ylabel="Pressure (bar)", xlim=(245, 335), ylim=(0, 120),
               title="CO₂ liquid–gas boundary predicted from gas-phase data")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    phase_figure = plot_phase_diagram()
    phase_figure
    return (phase_figure,)


if __name__ == "__main__":
    app.run()

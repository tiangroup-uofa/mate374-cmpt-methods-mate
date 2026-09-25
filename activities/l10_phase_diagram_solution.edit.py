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
    from scipy.interpolate import CubicSpline, PchipInterpolator

    return CubicSpline, PchipInterpolator, curve_fit, io, minimize_scalar, mo, np, plt, root_scalar


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## L10 · From CO₂ gas measurements to a liquid–gas boundary

    **Completed tutor reference.** All four live-code cells are filled in.
    Run them in sequence to fit the model, find both free-energy minima,
    solve for coexistence pressure, and interpolate the boundary. The collapsed
    cells supply measurements, search intervals, figures, and checks.

    **Engineering question.** At what pressure can liquid and gas CO₂ coexist
    between 250 and 295 K? We fit the van der Waals (vdW) model to all 24
    measurements below 10 mol/L, collected at 300–330 K.

    In $f(x_1,x_2;p_1,p_2)$, the semicolon separates variables from parameters
    held fixed for that calculation. Python separates all arguments with commas.
    """)
    return


@app.cell(hide_code=True)
def supplied_data(io, np):
    # Offline copy of data/L10-co2-pvt.csv: CO₂ isochoric (P, ρ, T) measurements
    # of Ely, Haynes and Bain (1989). Columns: T (K), P (MPa), ρ (mol/L).
    csv_text = """T_K,P_MPa,rho_mol_L
    300.0,2.1936,0.99431
    300.0,5.07784,2.99455
    305.0,7.33917,6.86619
    310.0,2.28675,0.99322
    310.0,5.43213,2.99001
    310.0,7.12164,4.98887
    310.0,7.86433,6.85824
    310.0,8.06822,7.83874
    310.0,8.20823,8.82938
    310.0,8.31622,9.87864
    310.0,8.44698,11.24963
    310.0,8.62861,12.59945
    310.0,9.02537,14.05004
    310.0,10.37431,15.95375
    320.0,2.37898,0.99211
    320.0,5.77614,2.98536
    320.0,7.79098,4.97857
    320.0,8.87599,6.84063
    320.0,9.26616,7.8168
    320.0,9.59111,8.80289
    320.0,9.89966,9.84698
    320.0,10.3143,11.21009
    320.0,10.82856,12.54955
    320.0,11.67866,13.98434
    320.0,13.81104,15.87138
    330.0,2.47023,0.991
    330.0,6.11169,2.97716
    330.0,8.43934,4.96743
    330.0,9.85303,6.81963
    330.0,10.42414,7.78929
    330.0,10.94096,8.76818
    330.0,11.46213,9.80457
    330.0,12.1827,11.15849
    330.0,13.0513,12.49189
    330.0,14.37347,13.92777
    330.0,17.32428,15.82941"""
    table = np.genfromtxt(io.StringIO(csv_text.replace("    ", "")), delimiter=",",
                          names=True, dtype=None, encoding="utf-8")
    T_data = table["T_K"].astype(float)            # K
    P_data = 10*table["P_MPa"].astype(float)       # bar
    rho_data = table["rho_mol_L"].astype(float)    # mol/L
    v_data = 1/rho_data                            # L/mol
    fit_set = rho_data < 10                       # All 24 states below 10 mol/L
    T_fit, v_fit, P_fit = T_data[fit_set], v_data[fit_set], P_data[fit_set]
    print(f"{fit_set.sum()} fitting states; largest density {rho_data[fit_set].max():.2f} mol/L")
    return P_data, P_fit, T_data, T_fit, fit_set, rho_data, v_data, v_fit


@app.cell(hide_code=True)
def supplied_constants():
    R = 0.08314462618  # L bar / (mol K)
    a_book, b_book = 3.592, 0.04267  # Course CO₂ parameters used in L05–L09

    def pressure_eos(v, T, a, b):
        # vdW pressure in bar; v in L/mol, T in K. Used by the supplied checks.
        return R*T/(v-b) - a/v**2

    return R, a_book, b_book, pressure_eos


@app.cell(hide_code=True)
def supplied_data_plot(P_data, T_data, mo, np, plt, rho_data):
    def plot_data():
        fig, ax = plt.subplots(figsize=(8, 5))
        for T in np.unique(T_data):
            on = T_data == T
            ax.scatter(rho_data[on], P_data[on], s=40, label=f"{T:g} K")
        ax.set(xlabel="Molar density (mol/L)", ylabel="Pressure (bar)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    data_figure = plot_data()
    mo.vstack([mo.md(r"""
    ### The measurements

    CO₂ measurements from Ely, Haynes and Bain (1989), *The Journal of Chemical
    Thermodynamics* **21**, 879–894. We use all 24 states with $\rho<10$ mol/L
    for fitting. The plot includes all 36 measurements, so we can also see
    the behaviour at higher density. Molar volume is $v=1/\rho$.
    """), data_figure])
    return (data_figure,)


@app.cell(hide_code=True)
def step_one(mo):
    mo.md(r"""
    ### Live step 1 · Fit $a$ and $b$ with `curve_fit`

    Model: $P_{\mathrm{vdW}}(T,v;a,b)=\dfrac{RT}{v-b}-\dfrac{a}{v^2}$.
    The arrays `T_fit`, `v_fit`, `P_fit` contain all 24 fitting states (K, L/mol, bar).
    Pass `(T_fit, v_fit)` as the known inputs and minimize squared pressure residuals.

    **Before coding:** Which quantities are measured, and which two does the fit change?

    ```python
    parameters, covariance = curve_fit(model, x_known, y_measured, p0=(guess_a, guess_b))
    # model(x_known, a, b): the known inputs come first, then the fitted parameters
    ```

    Docs: [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
    """)
    return


@app.cell
def live_fit(P_fit, R, T_fit, curve_fit, v_fit):
    def vdw_pressure(state, a, b):
        T, v = state
        return R*T/(v - b) - a/v**2

    (a_fit, b_fit), _ = curve_fit(vdw_pressure, (T_fit, v_fit), P_fit, p0=(3.6, 0.043))
    print(f"a = {a_fit:.5f} L² bar/mol²,  b = {b_fit:.6f} L/mol")
    return a_fit, b_fit, vdw_pressure


@app.cell(hide_code=True)
def check_fit(P_data, T_data, a_fit, b_fit, fit_set, mo, np, pressure_eos, v_data):
    mo.stop(a_fit is None or b_fit is None,
            mo.md("*Complete live step 1 to display the fitted pressure residuals.*"))
    pressure_residual = pressure_eos(v_data, T_data, a_fit, b_fit) - P_data
    fit_rmse = np.sqrt(np.mean(pressure_residual[fit_set]**2))
    dense_rmse = np.sqrt(np.mean(pressure_residual[~fit_set]**2))
    mo.md(f"""
    ### Pressure residuals

    | Density range | Number of states | RMS pressure residual (bar) |
    |---|---:|---:|
    | Below 10 mol/L (used for fitting) | {fit_set.sum()} | {fit_rmse:.2f} |
    | Above 10 mol/L | {(~fit_set).sum()} | {dense_rmse:.2f} |

    The fit should give approximately **a = 3.58867 L² bar/mol²** and
    **b = 0.042212 L/mol**, with an RMS pressure residual of **1.80 bar**
    in the fitted range. The much larger high-density discrepancy shows
    where this vdW approximation describes CO₂ poorly.
    """)
    return dense_rmse, fit_rmse


@app.cell(hide_code=True)
def supplied_well_limits(R, np, pressure_eos, root_scalar):
    def turning_volumes(T, a, b):
        # The two EOS turning points separate the liquid well (v < v1) from the gas well (v > v2).
        if not np.all(np.isfinite([T, a, b])) or min(T, a, b) <= 0:
            raise ValueError("T, a, and b must be finite and positive.")
        Tc = 8*a/(27*R*b)
        if T >= Tc:
            raise ValueError("No distinct liquid–gas coexistence at or above the model Tc.")
        if not 0.82 <= T/Tc <= 0.985:
            raise ValueError("This classroom scaffold supports 0.82 <= T/Tc <= 0.985.")

        # Solve dP/dv = 0 in the dimensionless volume w = v/b for numerical scaling.
        def slope(w):
            return -R*T*b/(a*(w-1)**2) + 2/w**3

        lower = root_scalar(slope, bracket=(1+1e-8, 3), method="brentq", xtol=1e-12, maxiter=100)
        upper = root_scalar(slope, bracket=(3, 2*a/(R*T*b)), method="brentq", xtol=1e-12, maxiter=100)
        if not lower.converged or not upper.converged:
            raise RuntimeError("The supplied turning-point calculation did not converge.")
        return b*lower.root, b*upper.root

    def find_pressure_bracket(T, a, b):
        # Pressures between the two turning-point pressures give two free-energy wells.
        v1, v2 = turning_volumes(T, a, b)
        low = max(pressure_eos(v1, T, a, b), 1e-3*a/(27*b*b))
        high = pressure_eos(v2, T, a, b)
        margin = 1e-4*(high-low)  # Keep two distinct wells at the endpoints.
        return low+margin, high-margin

    return find_pressure_bracket, turning_volumes


@app.cell(hide_code=True)
def step_two(mo):
    mo.md(r"""
    ### Live step 2 · Minimize each free-energy well with `minimize_scalar`

    At fixed $T,P$, minimize
    $\mathcal G(v)=-RT\ln[(v-b)/v_{\mathrm{ref}}]-a/v+Pv$ over $v$, with
    $v_{\mathrm{ref}}=1$ L/mol. Multiply L bar/mol by 100 to obtain J/mol.
    `turning_volumes(T, a, b)` locates the two zero-slope points of the isotherm:
    the liquid well lies in $(b,v_1)$ and the gas well beyond $v_2$.

    Write `find_state_volume` for one bounded search, then call it twice in
    `phase_volumes`. Bounded minimization keeps each search in its own well.
    **Predict:** As pressure increases at fixed temperature, which well becomes lower?

    ```python
    result = minimize_scalar(fun, bounds=(low, high), args=(T, P, a, b),
                             method="bounded", options={"xatol": 1e-10})
    result.x      # the volume at the bottom of the well
    ```

    Docs: [`scipy.optimize.minimize_scalar`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html)
    """)
    return


@app.cell
def live_minimize(R, minimize_scalar, np, turning_volumes):
    def free_energy(v, T, P, a, b):
        return 100*(-R*T*np.log(v - b) - a/v + P*v)  # J/mol

    def find_state_volume(T, P, a, b, bounds):
        result = minimize_scalar(free_energy, bounds=bounds, args=(T, P, a, b),
                                 method="bounded", options={"xatol": 1e-10})
        if not result.success:
            raise RuntimeError(result.message)
        return result.x

    def phase_volumes(T, P, a, b):
        v1, v2 = turning_volumes(T, a, b)
        v_liq = find_state_volume(T, P, a, b, bounds=(1.0001*b, v1))
        v_gas = find_state_volume(T, P, a, b, bounds=(v2, 5*R*T/P))
        return v_liq, v_gas

    return find_state_volume, free_energy, phase_volumes


@app.cell(hide_code=True)
def well_controls(a_fit, b_fit, find_pressure_bracket, mo, phase_volumes):
    mo.stop(a_fit is None or phase_volumes is None,
            mo.md("*Waiting for live steps 1 and 2.*"))
    well_T = 280.0  # K
    well_low, well_high = find_pressure_bracket(well_T, a_fit, b_fit)
    well_P = mo.ui.slider(float(round(well_low + 1)), float(round(well_high - 1)), step=0.1,
                          value=50.0, label="Trial pressure at 280 K (bar)",
                          show_value=True, debounce=True)
    well_P
    return well_P, well_T


@app.cell(hide_code=True)
def supplied_well_plot(a_fit, b_fit, free_energy, mo, np, phase_volumes, plt, pressure_eos, well_P, well_T):
    def plot_wells(P):
        g = lambda v: free_energy(v, well_T, P, a_fit, b_fit)
        v_liq, v_gas = phase_volumes(well_T, P, a_fit, b_fit)
        g_liq, g_gas = g(v_liq), g(v_gas)
        v = np.geomspace(1.2*b_fit, 2*v_gas, 600)
        curve = g(v)
        barrier = curve[(v > v_liq) & (v < v_gas)].max()
        low = min(g_liq, g_gas)
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.plot(v, curve - low, color="black", lw=1.2)
        ax.plot([v_liq], [g_liq - low], "o", color="tab:orange", ms=8, label=f"Liquid well: v = {v_liq:.4f} L/mol")
        ax.plot([v_gas], [g_gas - low], "o", color="tab:blue", ms=8, label=f"Gas well: v = {v_gas:.4f} L/mol")
        ax.set(xscale="log", xlabel="Molar volume v (L/mol, log scale)",
               ylabel="G − lower minimum (J/mol)", ylim=(-0.15*(barrier-low), 1.4*(barrier-low)),
               title=f"T = {well_T:g} K, P = {P:.1f} bar, fitted a and b")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        eos_error = max(abs(pressure_eos(v_liq, well_T, a_fit, b_fit) - P),
                        abs(pressure_eos(v_gas, well_T, a_fit, b_fit) - P))
        return fig, g_liq - g_gas, eos_error

    well_figure, well_gap, well_eos_error = plot_wells(well_P.value)
    minima_ok = well_eos_error < 1e-3
    mo.vstack([well_figure, mo.md(
        f"ΔG = G<sub>liq</sub> − G<sub>gas</sub> = **{well_gap:+.1f} J/mol**, so the "
        f"**{'liquid' if well_gap < 0 else 'gas'}** is favoured at this pressure."
    ), mo.callout(
        "Step 2 check passed: both minima satisfy the vdW equation to 1e-3 bar." if minima_ok else
        f"A minimum misses the vdW equation by {well_eos_error:.2g} bar: check the bounds and xatol.",
        kind="success" if minima_ok else "warn")])
    return


@app.cell(hide_code=True)
def step_three(mo):
    mo.md(r"""
    ### Live step 3 · Find the pressure where $\Delta G=0$ with `root_scalar`

    $\Delta G(P)=\mathcal G(v_{\mathrm{liq}})-\mathcal G(v_{\mathrm{gas}})$, with both volumes found again at each trial $P$.
    `find_pressure_bracket(T, a, b)` returns a pressure interval where $\Delta G$ changes sign.

    ```python
    result = root_scalar(f, args=(T, a, b), bracket=(low, high), method="bisect")
    result.root, result.converged
    # f(P, T, a, b): the unknown comes first, the fixed inputs follow in args
    ```

    Start with bisection, then try `method="brentq"` and compare the result.
    At 275 K, the fitted model gives approximately **50.0448 bar**.
    **Sanity check:** Where are the two volumes calculated when the root finder changes pressure?

    Docs: [`scipy.optimize.root_scalar`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html)
    """)
    return


@app.cell
def live_root(find_pressure_bracket, free_energy, phase_volumes, root_scalar):
    def delta_G(P, T, a, b):
        v_liq, v_gas = phase_volumes(T, P, a, b)
        return free_energy(v_liq, T, P, a, b) - free_energy(v_gas, T, P, a, b)

    def solve_pressure(T, a, b):
        result = root_scalar(delta_G, args=(T, a, b),
                             bracket=find_pressure_bracket(T, a, b), method="bisect")
        if not result.converged:
            raise RuntimeError("The coexistence-pressure solve did not converge.")
        return result.root

    return delta_G, solve_pressure


@app.cell(hide_code=True)
def check_trials(a_fit, b_fit, delta_G, mo, np, phase_volumes, pressure_eos, solve_pressure):
    mo.stop(a_fit is None or solve_pressure is None, mo.md("*Waiting for live step 3.*"))
    trial_T = [250.0, 270.0, 290.0]  # K
    trial_P = [solve_pressure(T, a_fit, b_fit) for T in trial_T]

    def check_trials():
        lines = ["| T (K) | P<sub>sat</sub> (bar) | ΔG (J/mol) | Largest EOS residual (bar) |",
                 "|---:|---:|---:|---:|"]
        checks = []
        for T, P in zip(trial_T, trial_P):
            volumes = np.array(phase_volumes(T, P, a_fit, b_fit))
            gap = delta_G(P, T, a_fit, b_fit)
            eos_error = np.max(np.abs(pressure_eos(volumes, T, a_fit, b_fit) - P))
            checks.append(abs(gap) < 1e-4 and eos_error < 1e-3)
            lines.append(f"| {T:g} | {P:.4f} | {gap:+.2e} | {eos_error:.2e} |")
        return all(checks), "\n".join(lines)

    checks_passed, trial_report = check_trials()
    mo.vstack([mo.md(trial_report), mo.callout(
        "Step 3 check passed." if checks_passed else
        "Check the pressure solve: require |ΔG| < 1e-4 J/mol and EOS residual < 1e-3 bar.",
        kind="success" if checks_passed else "warn",
    )])
    return checks_passed, trial_P, trial_T


@app.cell(hide_code=True)
def step_four(mo):
    mo.md(r"""
    ### Live step 4 · Interpolate five coexistence pressures

    Use `np.linspace(250, 295, 5)` to choose five temperatures, including both
    endpoints. Calculate their saturation pressures and pass the resulting
    arrays to the helper below.

    ```python
    boundary = interpolate_boundary(T_grid, P_grid, method="linear")
    boundary(275.0)   # evaluate pressure in bar; nan outside the sampled interval
    ```

    Start with `"linear"`, then try `"cubic"` and `"pchip"`. A cubic spline is smooth,
    while PCHIP preserves the increasing shape of these data. Compare each
    prediction at 275 K with a fresh coexistence calculation.
    A plotting grid spaced by 0.01 K gives a dense curve; its accuracy still
    depends on the five original calculations and the interpolation method.

    Docs: [`numpy.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) ·
    [`CubicSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html) ·
    [`PchipInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
    """)
    return


@app.cell(hide_code=True)
def supplied_interpolation(CubicSpline, PchipInterpolator, np):
    def interpolate_boundary(T_nodes, P_nodes, method="linear"):
        if method == "linear":
            return lambda T: np.interp(T, T_nodes, P_nodes, left=np.nan, right=np.nan)
        if method == "cubic":
            return CubicSpline(T_nodes, P_nodes, extrapolate=False)
        if method == "pchip":
            return PchipInterpolator(T_nodes, P_nodes, extrapolate=False)
        raise ValueError("Choose 'linear', 'cubic', or 'pchip'.")

    return (interpolate_boundary,)


@app.cell
def live_boundary(a_fit, b_fit, interpolate_boundary, np, solve_pressure):
    T_grid = np.linspace(250, 295, 5)  # K
    P_grid = [solve_pressure(T, a_fit, b_fit) for T in T_grid]
    boundary = interpolate_boundary(T_grid, P_grid, method="linear")
    return P_grid, T_grid, boundary


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
def supplied_answer(a_book, a_fit, b_book, b_fit, boundary, checks_passed, delta_G, measured_saturation_pressure, mo, np, solve_pressure):
    mo.stop(boundary is None or not checks_passed, mo.md("*Waiting for live step 4.*"))
    check_T = 275.0  # K, not one of the five sample temperatures
    direct_P = solve_pressure(check_T, a_fit, b_fit)
    interpolated_P = float(boundary(check_T))
    interpolated_gap = delta_G(interpolated_P, check_T, a_fit, b_fit)

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
        "| T (K) | vdW, fitted (bar) | vdW, course values (bar) | CO₂ reference (bar) | Fitted vs reference (%) |",
        "|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "The CO₂ reference is the vapour-pressure correlation of Span and Wagner (1996), "
        "https://doi.org/10.1063/1.555991. The fitted vdW model overpredicts saturation pressure, "
        "especially at lower temperatures. Small EOS and free-energy residuals check our "
        "numerical calculation; agreement with the reference data tests the physical model.",
    ]))
    return answer_T, answer_book, answer_fit, answer_measured, check_T, direct_P, interpolated_P


@app.cell(hide_code=True)
def supplied_phase_plot(P_data, P_grid, Pc_CO2, R, T_data, T_grid, Tc_CO2, a_book, a_fit, b_book, b_fit, boundary, check_T, direct_P, fit_set, measured_saturation_pressure, np, plt, solve_pressure):
    def plot_phase_diagram():
        grid = np.linspace(T_grid[0], T_grid[-1], 4501)  # 0.01 K spacing
        book_T = np.linspace(250.0, 295.0, 10)
        book_P = [solve_pressure(T, a_book, b_book) for T in book_T]
        measured_T = np.linspace(245.0, Tc_CO2, 300)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(measured_T, measured_saturation_pressure(measured_T), color="black",
                lw=2, label="CO₂ reference (Span–Wagner)")
        ax.plot(grid, boundary(grid), color="tab:red", lw=2, label="vdW, fitted a, b")
        ax.plot(book_T, book_P, color="tab:gray", ls="--", label="vdW, course a, b")
        ax.scatter(T_grid, P_grid, c="tab:red", s=30, zorder=4, label="Five coexistence solves")
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

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9", "scipy>=1.14"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def course_imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq, minimize_scalar

    return brentq, minimize_scalar, mo, np, plt


@app.cell(hide_code=True)
def energy_question(mo):
    mo.md(r"""
    ## L07 · Which CO₂ state has the lower free energy?
    **Predict:** click on the phase diagram to place your $(T, P)$ state — a short click drops a point,
    a small drag lets you fine-tune. Can you land on the coexistence line so the two free-energy minima balance?
    """)
    return


@app.cell(hide_code=True)
def energy_model(brentq, np):
    def co2_pressure(volume, temperature):
        # Input: L/mol, K. Output: bar.
        return 0.08314462618*temperature/(volume-0.04267) - 3.592/volume**2


    def co2_slope(volume, temperature):
        return -0.08314462618*temperature/(volume-0.04267)**2 + 2*3.592/volume**3


    def co2_energy(volume, temperature, pressure):
        # Convert L/mol to m³/mol before using the SI free-energy expression.
        v = np.asarray(volume) * 1e-3
        return -8.314462618*temperature*np.log((v-4.267e-5)/1e-3) - 0.3592/v + pressure*1e5*v


    def co2_volumes(temperature, pressure):
        # Independent reference: solve the cubic implied by the EOS.
        candidates = np.roots([pressure, -(pressure*0.04267+0.08314462618*temperature), 3.592, -3.592*0.04267])
        return np.sort([z.real for z in candidates if abs(z.imag) < 1e-8 and z.real > 0.04267])


    def co2_relative_energy(volume, temperature, pressure):
        # Use the gas-like minimum as zero in the two-well region.
        # With one minimum, use that sole minimum as the reference.
        reference_volume = co2_volumes(temperature, pressure)[-1]
        return co2_energy(volume, temperature, pressure) - co2_energy(reference_volume, temperature, pressure)


    def co2_saturation(temperature):
        # Spinodal pressures delimit the range with three stationary volumes.
        turning = np.roots([0.08314462618*temperature, -2*3.592, 4*3.592*0.04267, -2*3.592*0.04267**2])
        turning = sorted(z.real for z in turning if abs(z.imag)<1e-8 and z.real>0.04267)
        if len(turning) != 2:
            return np.nan
        low = max(1e-5, co2_pressure(turning[0], temperature))
        high = co2_pressure(turning[1], temperature)
        margin = (high-low)*1e-7
        def difference(pressure):
            volumes = co2_volumes(temperature, pressure)
            return co2_energy(volumes[0], temperature, pressure)-co2_energy(volumes[-1], temperature, pressure)
        return brentq(difference, low+margin, high-margin, xtol=1e-9)


    co2_Tc = 8*3.592/(27*0.08314462618*0.04267)
    co2_Pc = 3.592/(27*0.04267**2)
    co2_phase_T = np.linspace(250, co2_Tc-0.05, 45)
    co2_phase_P = np.array([co2_saturation(t) for t in co2_phase_T])
    return (
        co2_Pc,
        co2_Tc,
        co2_energy,
        co2_phase_P,
        co2_phase_T,
        co2_pressure,
        co2_relative_energy,
        co2_saturation,
        co2_slope,
        co2_volumes,
    )


@app.cell(hide_code=True)
def energy_controls(co2_Pc, co2_Tc, co2_phase_P, co2_phase_T, mo, np, plt):
    def draw_phase_picker():
        # Static phase diagram: the selection box you draw is your (T, P) marker.
        boundary_T = np.append(co2_phase_T, co2_Tc)
        boundary_P = np.append(co2_phase_P, co2_Pc)
        fig, phase = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
        phase.fill_between(boundary_T, 10, boundary_P, color="#1f77b4", alpha=0.14)
        phase.fill_between(boundary_T, boundary_P, 100, color="#ff7f0e", alpha=0.14)
        phase.fill_between([co2_Tc, 330], 10, co2_Pc, color="#1f77b4", alpha=0.14)
        phase.fill_between([co2_Tc, 330], co2_Pc, 100, color="#d62728", alpha=0.12)
        phase.plot(boundary_T, boundary_P, color="#d62728", lw=2, label="Liquid–gas coexistence")
        # Dotted guides mark the supercritical quadrant, not phase boundaries.
        phase.plot([co2_Tc, co2_Tc, 330], [100, co2_Pc, co2_Pc], color="0.5", ls=":", lw=1)
        phase.scatter(co2_Tc, co2_Pc, color="black", label="vdW critical point", s=30, zorder=4)
        phase.text(261, 65, "Liquid", color="#a84b00", fontsize=12, ha="center")
        phase.text(291, 23, "Gas", color="#165783", fontsize=12, ha="center")
        phase.text(315, 87, "Supercritical\n$T>T_c$, $P>P_c$", color="#a31d1e", fontsize=11, ha="center")
        phase.set(xlabel="Temperature (K)", ylabel="Pressure (bar)",
                  xlim=(250, 330), ylim=(10, 100), title="Phase diagram — click to set your (T, P)")
        phase.legend(fontsize=8, loc="upper left")
        phase.grid(alpha=0.2)
        return phase

    # Seed a small box centred on the previous default (280 K, 50 bar).
    energy_pick = mo.ui.matplotlib(draw_phase_picker(), value={"x": (279.0, 281.0), "y": (49.0, 51.0)}, debounce=True)
    energy_method = mo.ui.dropdown(["Bounded: each well", "Newton–Raphson: stationary point"], value="Bounded: each well", label="Solver")
    energy_guess = mo.ui.number(0.05, 3.0, step=0.01, value=0.30, label="Newton–Raphson starting volume (L/mol)")
    mo.vstack([energy_pick, mo.hstack([energy_method, energy_guess], wrap=True)])
    return energy_guess, energy_method, energy_pick


@app.cell(hide_code=True)
def energy_point(energy_pick):
    # Reduce the box selection to its centre: a quick click is a near-degenerate
    # box, so the centre is effectively the clicked point.
    _sel = energy_pick.value
    if _sel is None:
        energy_T, energy_P = 280.0, 50.0
    else:
        energy_T = min(max((_sel.x_min + _sel.x_max) / 2, 250.0), 330.0)
        energy_P = min(max((_sel.y_min + _sel.y_max) / 2, 10.0), 100.0)
    return energy_P, energy_T


@app.cell(hide_code=True)
def energy_solver(
    co2_energy,
    co2_pressure,
    co2_slope,
    co2_volumes,
    energy_P,
    energy_T,
    energy_guess,
    energy_method,
    minimize_scalar,
    np,
):
    def solve_co2_minima(temperature, pressure, method, guess):
        stationary = co2_volumes(temperature, pressure)
        upper = max(1.0, 2*stationary[-1])
        rows = []
        if method == "Bounded: each well":
            # Split the search at the independently located maximum when it exists.
            intervals = [(0.04267+1e-6, upper)]
            if len(stationary) == 3:
                intervals = [(0.04267+1e-6, stationary[1]), (stationary[1], upper)]
            for left, right in intervals:
                result = minimize_scalar(lambda v: co2_energy(v, temperature, pressure), bounds=(left,right), method="bounded", options={"xatol":1e-11})
                rows.append(dict(volume=result.x, energy=float(result.fun), iterations=int(result.nit), calls=int(result.nfev), success=bool(result.success)))
            message = "Each interval contains one well; the middle stationary volume splits the two-well case."
        else:
            # Transparent Newton–Raphson update on dG/dv = 100(P_external − P_EOS).
            volume = float(guess)
            success = False
            message = "Iteration limit"
            for k in range(40):
                slope = co2_slope(volume, temperature)
                if abs(slope) < 1e-10:
                    message = "Stopped: curvature too small"
                    break
                trial = volume + (pressure-co2_pressure(volume, temperature))/slope
                if not np.isfinite(trial) or trial <= 0.04267:
                    message = "Stopped: Newton–Raphson step left the physical domain"
                    break
                small_step = abs(trial-volume) <= 1e-8
                volume = trial
                if small_step and abs(co2_pressure(volume, temperature)-pressure) <= 1e-6:
                    success = True
                    message = "Stationary point found; use curvature to classify it"
                    break
            rows.append(dict(volume=volume, energy=float(co2_energy(volume,temperature,pressure)), iterations=k+1, calls=2*(k+1), success=success))
        for row in rows:
            row["pressure_residual"] = co2_pressure(row["volume"],temperature)-pressure
            curvature = -100*co2_slope(row["volume"],temperature)
            row["kind"] = "minimum" if curvature>0 else "maximum" if curvature<0 else "flat"
        return stationary, rows, message


    energy_stationary, energy_rows, energy_message = solve_co2_minima(energy_T, energy_P, energy_method.value, energy_guess.value)
    return energy_message, energy_rows, energy_stationary, solve_co2_minima


@app.cell(hide_code=True)
def energy_plot(
    co2_pressure,
    co2_relative_energy,
    energy_P,
    energy_T,
    energy_message,
    energy_rows,
    energy_stationary,
    mo,
    np,
    plt,
):
    def draw_co2_landscape(temperature, pressure, stationary, rows):
        upper = max(0.35, 1.35*stationary[-1])
        volumes = np.linspace(0.050, upper, 1800)
        fig = plt.figure(figsize=(10, 4.2), layout="constrained")
        grid = fig.add_gridspec(1, 2)
        eos = fig.add_subplot(grid[0, 0])
        gibbs = fig.add_subplot(grid[0, 1])

        eos.plot(volumes, co2_pressure(volumes, temperature), color="#1f77b4")
        eos.axhline(pressure, color="0.3", ls="--", label=f"P = {pressure:.2f} bar")
        eos.scatter(stationary, np.full(len(stationary), pressure), color="black", s=24)
        eos.set(xlabel="Molar volume (L/mol)", ylabel="Pressure (bar)",
                xlim=(0.050, upper), ylim=(0, 110), title="van der Waals EOS")
        eos.legend(fontsize=8)

        energy_values = co2_relative_energy(stationary, temperature, pressure)
        gibbs.plot(volumes, co2_relative_energy(volumes, temperature, pressure), color="#ff7f0e")
        gibbs.axhline(0, color="0.6", lw=0.8)
        gibbs.scatter(stationary, energy_values, color="black", s=25, label="Stationary reference")
        for index, row in enumerate(rows):
            gibbs.scatter(row["volume"], co2_relative_energy(row["volume"], temperature, pressure),
                          marker="x", color="#d62728", s=65, zorder=4,
                          label="Solver result" if index == 0 else None)
        reference = r"G_{\mathrm{gas}}" if len(stationary) == 3 else r"G_{\mathrm{min}}"
        gibbs.set(xlabel="Molar volume (L/mol)",
                  ylabel=rf"$\mathcal{{G}}(v;T,P)-{reference}(T,P)$ (J/mol)",
                  xlim=(0.050, upper), ylim=(min(energy_values)-30, max(energy_values)+80),
                  title="Gibbs free energy and minimum")
        gibbs.legend(fontsize=8)
        for ax in (eos, gibbs):
            ax.grid(alpha=0.2)
        fig.suptitle(f"vdW CO₂ at your selected state:  T = {temperature:g} K,  P = {pressure:.2f} bar")
        return fig


    energy_figure = draw_co2_landscape(energy_T, energy_P, energy_stationary, energy_rows)
    energy_report = "\n".join(
        f"- v = **{row['volume']:.7f} L/mol**, relative G = **{co2_relative_energy(row['volume'], energy_T, energy_P):.3f} J/mol**; {row['kind']}; "
        f"solver success: **{row['success']}**; pressure residual **{row['pressure_residual']:+.2g} bar**; **{row['iterations']} iterations**."
        for row in energy_rows)
    if len(energy_stationary) == 3:
        energy_difference = float(co2_relative_energy(energy_stationary[0], energy_T, energy_P))
        if abs(energy_difference) <= 0.1:
            energy_preference = "The minima are balanced to within 0.1 J/mol."
        elif energy_difference < 0:
            energy_preference = "The small-volume minimum is lower."
        else:
            energy_preference = "The large-volume minimum is lower."
        energy_interpretation = f"G(liquid) − G(gas) = **{energy_difference:+.3f} J/mol**. " + energy_preference
    else:
        energy_interpretation = "There is one minimum, used as the energy zero. A single minimum can occur below Tc too, outside the three-root pressure range."
    mo.vstack([
        energy_figure,
        mo.md(energy_report + "\n\n" + energy_interpretation + "\n\n" + energy_message),
        mo.accordion({"What should I check?": mo.md(
            "The gas-like minimum sets the energy zero when two wells exist, just as in the lecture figure. Near **280 K**, click just below then just above the red coexistence line and watch which well drops lower. Landing exactly on it balances the two minima — a small drag lets you nudge the point. Try Newton–Raphson at **0.11 L/mol**: can a successful root solve find a maximum? Click into the supercritical corner (**T > 304 K, P > 74 bar**) and watch the two wells merge into one. Compare the two minima only at the same T and P.\n\n"
            "A minimum with higher free energy is metastable in the homogeneous model. The intervening maximum signals an unstable volume; predicting a nucleation rate also requires interfacial physics. Newton–Raphson evaluates pressure and its derivative each iteration; the bounded solver evaluates free energy, so iteration counts measure different work." )}),
    ])
    return draw_co2_landscape, energy_figure


if __name__ == "__main__":
    app.run()

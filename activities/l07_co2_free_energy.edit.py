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
    **Predict:** at 280 K, will raising the pressure favour the smaller-volume or larger-volume minimum?

    Use molar volume $v$ and the dimensional vdW free-energy landscape
    $\mathcal G(v;T,P)=-RT\ln[(v-b)/v_{
    m ref}]-a/v+Pv$.
    Here $a=0.3592$ Pa m⁶ mol⁻², $b=4.267	imes10^{-5}$ m³ mol⁻¹, and $v_{
    m ref}=10^{-3}$ m³ mol⁻¹. The temperature-only constant is set to zero; it cancels when comparing phases at the same T and P. Plots use L/mol, bar, and J/mol.
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
        co2_slope,
        co2_volumes,
    )


@app.cell(hide_code=True)
def energy_controls(mo):
    energy_T = mo.ui.slider(250, 330, step=1, value=280, label="Temperature (K)", show_value=True, debounce=True)
    energy_P = mo.ui.slider(10, 100, step=0.5, value=50, label="Pressure (bar)", show_value=True, debounce=True)
    energy_method = mo.ui.dropdown(["Bounded: each well", "Newton: stationary point"], value="Bounded: each well", label="Solver")
    energy_guess = mo.ui.number(0.05, 3.0, step=0.01, value=0.30, label="Newton starting volume (L/mol)")
    mo.vstack([mo.hstack([energy_T, energy_P], widths="equal"), mo.hstack([energy_method, energy_guess], wrap=True)])
    return energy_P, energy_T, energy_guess, energy_method


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
            # Transparent Newton update on dG/dV = 100(P_external − P_EOS).
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
                    message = "Stopped: Newton step left the physical domain"
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


    energy_stationary, energy_rows, energy_message = solve_co2_minima(energy_T.value, energy_P.value, energy_method.value, energy_guess.value)
    return energy_message, energy_rows, energy_stationary


@app.cell(hide_code=True)
def energy_plot(
    co2_Pc,
    co2_Tc,
    co2_energy,
    co2_phase_P,
    co2_phase_T,
    co2_pressure,
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
        upper = max(0.65, 1.35*stationary[-1])
        volumes = np.linspace(0.050, upper, 1800)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained", gridspec_kw={"width_ratios":[1,1.2,1]})
        axes[0].plot(volumes, co2_pressure(volumes,temperature), color="#007c41")
        axes[0].axhline(pressure, color="#d87700", ls="--")
        axes[0].scatter(stationary, np.full(len(stationary), pressure), color="black", s=24)
        axes[0].set(xlabel="Molar volume (L/mol)", ylabel="Pressure (bar)", ylim=(0,110), title="EOS: possible volumes")
        axes[1].plot(volumes, co2_energy(volumes,temperature,pressure), color="#007c41")
        axes[1].scatter(stationary, co2_energy(stationary,temperature,pressure), color="black", s=25, label="Stationary reference")
        for row in rows:
            axes[1].scatter([row["volume"]], [row["energy"]], marker="x", color="#d87700", s=65)
        energy_values = co2_energy(stationary,temperature,pressure)
        axes[1].set(xlabel="Molar volume (L/mol)", ylabel="Molar free energy (J/mol)", title="Lower minimum is favoured", ylim=(min(energy_values)-30, max(energy_values)+250))
        axes[1].legend(fontsize=7)
        axes[2].plot(co2_phase_T, co2_phase_P, color="#007c41", label="Equal-minimum pressure")
        axes[2].scatter([co2_Tc], [co2_Pc], color="black", label="vdW critical point", s=25)
        axes[2].scatter([temperature], [pressure], color="#d87700", s=65, label="Your T, P")
        axes[2].set(xlabel="Temperature (K)", ylabel="Pressure (bar)", xlim=(250,330), ylim=(10,100), title="vdW CO₂ liquid–gas boundary")
        axes[2].legend(fontsize=7)
        for ax in axes:
            ax.grid(alpha=0.2)
        return fig


    energy_figure = draw_co2_landscape(energy_T.value, energy_P.value, energy_stationary, energy_rows)
    energy_report = "\n".join(
        f"- V = **{r['volume']:.7f} L/mol**, G = **{r['energy']:.3f} J/mol**; {r['kind']}; "
        f"solver success: **{r['success']}**; pressure residual **{r['pressure_residual']:+.2g} bar**; **{r['iterations']} iterations**."
        for r in energy_rows)
    if len(energy_stationary) == 3:
        energy_difference = float(co2_energy(energy_stationary[0],energy_T.value,energy_P.value)-co2_energy(energy_stationary[-1],energy_T.value,energy_P.value))
        energy_interpretation = f"G(small-volume) − G(large-volume) = **{energy_difference:+.3f} J/mol**. " + ("The small-volume minimum is lower." if energy_difference < 0 else "The large-volume minimum is lower.")
    else:
        energy_interpretation = "There is one stationary volume at these conditions. A single minimum can occur below Tc too, outside the three-root pressure range."
    mo.vstack([
        energy_figure,
        mo.md(energy_report + "\n\n" + energy_interpretation + "\n\n" + energy_message),
        mo.accordion({"What should I check?": mo.md(
            "At **280 K**, compare **50** and **56 bar**. Try Newton at **0.11 L/mol**: can a successful root solve find a maximum? Raise T to **310 K** and vary P. Compare the two minima only at the same T and P. The plotted boundary is calculated from this vdW CO₂ model; quantitative comparison with measured CO₂ needs validation.\n\n"
            "A minimum with higher free energy is metastable in the homogeneous model. The intervening maximum signals an unstable volume; predicting a nucleation rate also requires interfacial physics. Newton evaluates pressure and its derivative each iteration; the bounded solver evaluates free energy, so iteration counts measure different work." )}),
    ])
    return


if __name__ == "__main__":
    app.run()

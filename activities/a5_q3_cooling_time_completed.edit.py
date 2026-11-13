# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # A5 Q3 · Cooling time of a steel sphere

    A steel sphere cools by radiation according to the Stefan–Boltzmann equation,
    where $T_\mathrm a$ is the wall temperature:

    $$
    \frac{\mathrm dT}{\mathrm dt}=-\alpha(T^4-T_\mathrm a^4).
    $$

    We want to calculate the time $t_\mathrm f$ needed to reach a final temperature
    $T_\mathrm f$. We will compare two methods:

    **1. Integration.** Rearranging the ODE gives

    $$
    t_\mathrm f=\int_{T_\mathrm f}^{1200\ \mathrm K}
    \frac{\mathrm dT}{\alpha(T^4-T_\mathrm a^4)}.
    $$

    This is a **numerical integration** task.

    **2. ODE solution.** The explicit Euler method advances the temperature by

    $$
    T_{i+1}=T_i-h\alpha(T_i^4-T_\mathrm a^4).
    $$

    Complete **Step 1** for the integral and **Step 2** for explicit Euler.
    """)
    return


@app.cell
def parameters():
    T_0 = 1200.0              # initial temperature (K)
    T_a = 300.0               # wall temperature (K)
    alpha = 3.49e-11          # K^-3 s^-1
    return T_0, T_a, alpha


@app.cell(hide_code=True)
def inputs(mo):
    T_final = mo.ui.number(
        start=300, stop=1199.9, step=0.1, value=600,
        label="Final temperature T_f (K)",
    )
    h_step = mo.ui.number(
        start=0, stop=20, step=0.01, value=5,
        label="Euler time step h (s)",
    )
    mo.hstack([T_final, h_step], align="start", justify="start")
    return T_final, h_step


@app.cell(hide_code=True)
def step_one_text(mo):
    mo.md(r"""
    ### **Step 1** · Cooling time from the integral

    Complete the integrand in the cell below. The call to `quad` integrates it from
    $T_\mathrm f$ to the initial temperature $T_0$. It returns the cooling time
    and an estimate of the numerical integration error. Choose $T_\mathrm f>300$ K:
    the time to reach the wall temperature itself is infinite.

    ##### Why do we use `quad`?

    The integrand is an analytical function that we can evaluate at any temperature.
    `quad` selects its evaluation points adaptively, without a fixed table of values.
    Simpson's rule can also integrate sampled values on a suitable grid. Both methods
    perform numerical integration. See the integration-method chart in
    [Lecture 20](https://tiangroup-uofa.github.io/mate374-cmpt-methods-mate/units/04/L20/index.html#numerical-integration-in-python).
    """)
    return


@app.cell
def step_one(T_0, T_a, T_final, alpha, np):
    #######################################
    # Step 1: your move
    #         implement the integrand
    #######################################
    from scipy.integrate import quad

    def integrand(T):
        # Step 1: replace np.nan with the integrand
        return 1 / (alpha * (T**4 - T_a**4))

    # quad evaluates the function directly, without a sampled array.
    # It returns the integral (t_quad) and an estimated numerical error.
    t_quad, error_quad = quad(integrand, a=T_final.value, b=T_0)
    return error_quad, integrand, t_quad


@app.cell(hide_code=True)
def quad_diagnostics(T_0, T_a, T_final, alpha, integrand, np, t_quad):
    from scipy.integrate import quad as _quad

    reference_time = np.nan
    quad_evaluations = None
    if T_final.value is not None and T_a < T_final.value < T_0:
        def _reference_integrand(T):
            return 1 / (alpha * (T**4 - T_a**4))

        # Tighter-tolerance comparison for the stated cooling equation.
        reference_time = _quad(
            _reference_integrand, T_final.value, T_0, epsabs=1e-10, epsrel=1e-12
        )[0]
        if np.isfinite(t_quad):
            # Repeat the same quad call with diagnostics to obtain its evaluation count.
            _details = _quad(integrand, T_final.value, T_0, full_output=1)
            quad_evaluations = _details[2]["neval"]

    return quad_evaluations, reference_time


@app.cell(hide_code=True)
def integrand_check(T_0, T_a, T_final, alpha, mo, np, t_quad):
    if not np.isfinite(t_quad):
        _out = mo.callout(
            mo.md("**Wrong integrand function definition!** Check again."), kind="danger"
        )
    elif T_final.value <= T_a or T_final.value >= T_0:
        _out = mo.callout(
            mo.md("Choose a final temperature above the wall temperature and below the initial temperature."),
            kind="danger",
        )
    else:
        # Independent analytical antiderivative of 1/[alpha*(T^4 - T_a^4)].
        def _antiderivative(T):
            return (np.log((T - T_a) / (T + T_a)) - 2 * np.arctan(T / T_a)) / (
                4 * alpha * T_a**3
            )

        _expected = _antiderivative(T_0) - _antiderivative(T_final.value)
        if np.isclose(t_quad, _expected, rtol=1e-6, atol=1e-8):
            _out = mo.callout(mo.md("**Integrand check passed.**"), kind="success")
        else:
            _out = mo.callout(
                mo.md("**The integral is not correct.** Check your integrand and integration limits."),
                kind="danger",
            )
    _out
    return


@app.cell(hide_code=True)
def integral_result(error_quad, mo, np, quad_evaluations, t_quad):
    if np.isfinite(t_quad):
        _out = mo.md(f"**Cooling time from quad:** {t_quad:.4f} s "
                     f"(estimated numerical error: {error_quad:.2g} s; "
                     f"function evaluations: {quad_evaluations})")
    else:
        _out = mo.md("**Cooling time from quad:** not available yet.")
    _out
    return


@app.cell(hide_code=True)
def step_two_text(mo):
    mo.md(r"""
    ### **Step 2** · Cooling time by explicit Euler

    Complete `euler(T, h)` using the Euler formula above. The supplied function
    `time_to_cool(T_f, h)` calls it repeatedly and records the temperature until
    the sphere reaches $T_\mathrm f$ or below.
    """)
    return


@app.cell
def cooling_simulation(T_0, T_a, euler, np):
    def time_to_cool(T_f, h):
        times = [0.0]
        temperatures = [T_0]
        # Stop after 5000 s or 200,000 steps if the target is not reached.
        while temperatures[-1] > T_f and times[-1] < 5000 and len(times) <= 200000:
            if h <= 0:
                break
            T_next = euler(temperatures[-1], h)
            times.append(times[-1] + h)
            temperatures.append(T_next)
            if not np.isfinite(T_next) or T_next >= temperatures[-2] or T_next < T_a:
                break
        return np.array(times), np.array(temperatures)

    return (time_to_cool,)


@app.cell
def euler_step(T_a, alpha):
    #######################################
    # Step 2: your move
    #         implement the Euler step
    #######################################
    def euler(T, h):
        # Replace T with the explicit Euler update.
        return T - h * alpha * (T**4 - T_a**4)

    return (euler,)


@app.cell(hide_code=True)
def euler_plot(T_0, T_a, T_final, alpha, h_step, mo, np, plt, time_to_cool):
    _valid_inputs = (T_final.value is not None and T_a < T_final.value < T_0
                     and h_step.value is not None and h_step.value > 0)
    _times, _temps = (time_to_cool(T_final.value, h_step.value) if _valid_inputs
                      else (np.array([0.0]), np.array([T_0])))

    _status = ""
    _reached = False
    if not _valid_inputs:
        _status = "Choose T_a < T_f < T_0 and h > 0."
    elif len(_times) < 2 or not np.all(np.isfinite(_temps)):
        _status = "Invalid Euler update: check the temperature calculation."
    else:
        _expected = _temps[:-1] - h_step.value * alpha * (_temps[:-1]**4 - T_a**4)
        if not np.allclose(_temps[1:], _expected, rtol=1e-6, atol=1e-8):
            _status = "Incorrect Euler update: check your formula."
        elif np.any(_temps < T_a):
            _status = "Temperature fell below the wall temperature: reduce h."
        elif _temps[-1] > T_final.value:
            _status = "Target not reached within the time or step limit."
        else:
            _reached = True

    fig, ax = plt.subplots(figsize=(6.5, 3.5), layout="constrained")
    ax.plot(_times, _temps, "o-", color="tab:blue", ms=2.5, lw=1, label="Explicit Euler")
    ax.axhline(T_a, color="0.6", ls=":", label="Wall temperature")
    if T_final.value is not None:
        ax.axhline(T_final.value, color="0.35", ls="--", label="Final temperature")
    ax.set(xlabel="Time (s)", ylabel="Temperature (K)")
    ax.legend(frameon=False, fontsize=8)
    if _reached:
        ax.text(0.04, 0.05, f"Cooling time: {_times[-1]:.2f} s",
                transform=ax.transAxes, ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.95))
    if _status:
        ax.text(0.5, 0.93, _status, color="firebrick", weight="bold",
                ha="center", va="top", transform=ax.transAxes, wrap=True,
                bbox=dict(facecolor="white", edgecolor="firebrick", alpha=0.95))
    euler_reached, euler_temps, euler_times = _reached, _temps, _times
    mo.vstack([h_step, fig])
    return euler_reached, euler_temps, euler_times


@app.cell(hide_code=True)
def comparison_text(mo):
    mo.md(r"""
    ### Compare the two methods

    Set $T_\mathrm f$ and the Euler time step $h$. Compare the cooling times,
    function evaluations, and absolute errors. Which method gives the more precise
    cooling time? The errors are differences from a `quad` calculation with tighter
    tolerances. Each Euler step evaluates `euler(T, h)` once.
    """)
    return


@app.cell(hide_code=True)
def results(T_final, euler_reached, euler_times, h_step, mo, np,
            quad_evaluations, reference_time, t_quad):
    _quad = f"{t_quad:.4f}" if np.isfinite(t_quad) else "Complete Step 1"
    _quad_count = str(quad_evaluations) if quad_evaluations is not None else "—"
    _quad_error = (f"{abs(t_quad - reference_time):.3g}"
                   if np.isfinite(t_quad) and np.isfinite(reference_time) else "—")
    _euler = f"{euler_times[-1]:.4f}" if euler_reached else "Not available"
    _euler_count = str(len(euler_times) - 1) if euler_reached else "—"
    _euler_error = (f"{abs(euler_times[-1] - reference_time):.3g}"
                    if euler_reached and np.isfinite(reference_time) else "—")
    mo.vstack([
        mo.hstack([T_final, h_step], align="start", justify="start"),
        mo.md("| Method | Cooling time (s) | Function evaluations | Absolute error (s) |\n"
              "|---|---:|---:|---:|\n"
              f"| `quad` | {_quad} | {_quad_count} | {_quad_error} |\n"
              f"| Explicit Euler | {_euler} | {_euler_count} | {_euler_error} |"),
    ])
    return


if __name__ == "__main__":
    app.run()

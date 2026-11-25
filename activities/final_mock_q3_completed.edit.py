# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy", "scipy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Mock final · Q3 · completed")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Q3 · Cooling a steel plate

    ### (a) Long-time temperature

    The plate cools toward 300 K. The model gives zero cooling rate there
    and a negative rate above 300 K.

    ### (b) Cooling time

    The supplied forward Euler function estimates the temperature at a trial
    time. A root finder varies that time until the estimate equals 350 K.
    """)
    return


@app.cell
def _():
    def final_temp(t_final, n):
        """Return the plate temperature (K) after n Euler steps to t_final (s)."""
        h = t_final / n
        temperature = 1200.0
        for _ in range(n):
            temperature += h * (-0.2 * temperature + 60.0)
        return temperature

    return (final_temp,)


@app.cell
def _(final_temp):
    from scipy.optimize import brentq

    def residual(t_final, n):
        return final_temp(t_final, n) - 350.0

    for steps in (100, 1000):
        cooling_time = brentq(residual, 10.0, 20.0, args=(steps,))
        print(f"Euler + root finding ({steps} steps): {cooling_time:.5f} s")
    return


@app.cell
def _():
    from math import log

    # Exact solution: T(t) = 300 + 900 exp(-0.2 t).
    exact_time = log(900.0 / 50.0) / 0.2
    print(f"Analytical crossing: {exact_time:.5f} s")
    return


@app.cell
def _():
    from scipy.integrate import solve_ivp

    def cooling_rate(t, state):
        return [-0.2 * state[0] + 60.0]

    def crossing(t, state):
        return state[0] - 350.0

    crossing.direction = -1
    crossing.terminal = True
    solution = solve_ivp(
        cooling_rate,
        (0.0, 20.0),
        [1200.0],
        events=crossing,
        rtol=1e-8,
        atol=1e-8,
    )
    print(f"Runge–Kutta crossing: {solution.t_events[0][0]:.5f} s")
    return


if __name__ == "__main__":
    app.run()

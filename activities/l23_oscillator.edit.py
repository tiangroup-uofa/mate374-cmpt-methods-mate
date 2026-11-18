# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    method = mo.ui.dropdown(
        ["Euler", "Midpoint RK2", "Heun RK2", "RK4"], value="Euler", label="Method"
    )
    step = mo.ui.dropdown(
        options={"0.2": 0.2, "0.1": 0.1, "0.05": 0.05, "0.025": 0.025},
        value="0.2", label="Time step h"
    )
    displacement = mo.ui.slider(-1.0, 1.0, step=0.25, value=1.0, label="q(0)", show_value=True)
    velocity = mo.ui.slider(-1.0, 1.0, step=0.25, value=0.0, label="v(0)", show_value=True)
    mo.vstack([mo.hstack([method, step]), mo.hstack([displacement, velocity])])
    return displacement, method, step, velocity


@app.cell
def _(np):
    # Normalized mass and stiffness: m = k = 1, so omega = 1.
    def rhs(t, state):
        q, v = state
        return np.array([v, -q])

    def euler_step(t, state, h):
        return state + h * rhs(t, state)

    def midpoint_step(t, state, h):
        k1 = rhs(t, state)
        k2 = rhs(t + h / 2, state + h * k1 / 2)
        return state + h * k2

    def heun_step(t, state, h):
        k1 = rhs(t, state)
        k2 = rhs(t + h, state + h * k1)
        return state + h * (k1 + k2) / 2

    def rk4_step(t, state, h):
        k1 = rhs(t, state)
        k2 = rhs(t + h / 2, state + h * k1 / 2)
        k3 = rhs(t + h / 2, state + h * k2 / 2)
        k4 = rhs(t + h, state + h * k3)
        return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return euler_step, heun_step, midpoint_step, rhs, rk4_step


@app.cell
def _(displacement, euler_step, heun_step, method, midpoint_step, np, rk4_step, step, velocity):
    advance = {
        "Euler": euler_step,
        "Midpoint RK2": midpoint_step,
        "Heun RK2": heun_step,
        "RK4": rk4_step,
    }[method.value]
    h = step.value
    initial_state = np.array([displacement.value, velocity.value])
    times = np.linspace(0.0, 10.0, round(10.0 / h) + 1)
    states = np.empty((len(times), 2))
    states[0] = initial_state

    for n in range(len(times) - 1):
        states[n + 1] = advance(times[n], states[n], h)

    exact = np.column_stack((
        initial_state[0] * np.cos(times) + initial_state[1] * np.sin(times),
        -initial_state[0] * np.sin(times) + initial_state[1] * np.cos(times),
    ))
    energy = 0.5 * np.sum(states**2, axis=1)
    initial_energy = 0.5 * np.sum(initial_state**2)
    return energy, exact, h, initial_energy, initial_state, states, times


@app.cell(hide_code=True)
def _(energy, exact, initial_energy, method, mo, np, plt, states, times):
    with plt.rc_context({"font.size": 12}):
        figure, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True, layout="constrained")
        axes[0].plot(times, exact[:, 0], color="0.25", label="Exact")
        axes[0].plot(times, states[:, 0], "--", color="#b5473a", label=method.value)
        axes[0].set_ylabel("Displacement q")
        axes[0].legend(ncol=2)
        axes[1].plot(times, states[:, 0] - exact[:, 0], color="#b5473a")
        axes[1].axhline(0, color="0.5", lw=0.7)
        axes[1].set_ylabel("Error in q")
        axes[2].plot(times, energy, color="#b5473a")
        axes[2].axhline(initial_energy, color="0.25", ls="--")
        axes[2].set(xlabel="Time (normalized)", ylabel="Total energy E")
    mo.vstack([
        figure,
        mo.md(
            f"**At t = 10:** q = {states[-1, 0]:.6f}, v = {states[-1, 1]:.6f}; "
            f"state error = {np.linalg.norm(states[-1] - exact[-1]):.6g}; "
            f"E = {energy[-1]:.6f} (initial E = {initial_energy:.6f})."
        ),
    ])
    return


if __name__ == "__main__":
    app.run()

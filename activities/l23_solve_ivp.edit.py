# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from scipy.integrate import solve_ivp

    return np, solve_ivp


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    q0 = mo.ui.slider(-1.0, 1.0, step=0.25, value=1.0, label="Initial displacement q(0)", show_value=True)
    v0 = mo.ui.slider(-1.0, 1.0, step=0.25, value=0.0, label="Initial velocity v(0)", show_value=True)
    mo.hstack([q0, v0])
    return mo, plt, q0, v0


@app.cell
def _(np, q0, solve_ivp, v0):
    def oscillator(t, y):
        q, v = y
        return np.array([v, -q])  # Return dq/dt, then dv/dt.

    result = solve_ivp(
        fun=oscillator,
        t_span=(0.0, 10.0),
        y0=[q0.value, v0.value],
        method="RK45",
        t_eval=np.linspace(0.0, 10.0, 201),
        rtol=1e-8,
        atol=1e-10,
    )
    assert result.success, result.message
    return oscillator, result


@app.cell(hide_code=True)
def _(mo, plt, result):
    with plt.rc_context({"font.size": 12}):
        figure, axis = plt.subplots(figsize=(7, 3), layout="constrained")
        axis.plot(result.t, result.y[0], label="Displacement q")
        axis.plot(result.t, result.y[1], label="Velocity v")
        axis.set(xlabel="Time (normalized)", ylabel="State components (normalized)")
        axis.legend(ncol=2)
    mo.vstack([
        figure,
        mo.md(
            f"**At t = 10:** q = {result.y[0, -1]:.6f}, "
            f"v = {result.y[1, -1]:.6f}. "
            f"**result.y shape:** {result.y.shape}."
        ),
    ])
    return


if __name__ == "__main__":
    app.run()

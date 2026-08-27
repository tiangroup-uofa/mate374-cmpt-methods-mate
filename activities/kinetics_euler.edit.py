# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # From a materials inventory to an ODE

    Let $\alpha(t)$ be the fraction of a material transformed by a simple
    first-order mechanism. If untransformed material is available, take

    $$
    \frac{\mathrm{d}\alpha}{\mathrm{d}t}=k(1-\alpha),
    \qquad \alpha(0)=0.
    $$

    The exact solution is $\alpha(t)=1-\exp(-kt)$. Euler's method replaces
    the derivative with a finite step:

    $$
    \alpha_{n+1}=\alpha_n+\Delta t\,k(1-\alpha_n).
    $$

    This is the same course loop again: state variable, model, numerical
    representation, computation, verification, and physical interpretation.
    """)
    return


@app.cell
def _(mo):
    rate = mo.ui.slider(
        start=0.005,
        stop=0.100,
        step=0.005,
        value=0.020,
        label="Transformation rate k (1/min)",
        show_value=True,
    )
    final_time = mo.ui.slider(
        start=20,
        stop=200,
        step=10,
        value=100,
        label="Final time (min)",
        show_value=True,
    )
    steps = mo.ui.slider(
        start=5,
        stop=100,
        step=5,
        value=25,
        label="Euler steps",
        show_value=True,
    )
    mo.vstack([rate, final_time, steps], align="start", gap=1)
    return final_time, rate, steps


@app.cell
def _(np):
    def euler_transformation(rate, final_time, n_steps):
        # Given function: edit the update rule, not the interface.
        time = np.linspace(0.0, final_time, n_steps + 1)
        fraction = np.zeros(n_steps + 1)
        step = final_time / n_steps
        for index in range(n_steps):
            fraction[index + 1] = fraction[index] + step * rate * (
                1.0 - fraction[index]
            )
        return time, fraction

    return (euler_transformation,)


@app.cell
def _(euler_transformation, final_time, np, rate, steps):
    time, fraction = euler_transformation(
        rate.value, final_time.value, int(steps.value)
    )
    exact_fraction = 1.0 - np.exp(-rate.value * time)
    final_error = fraction[-1] - exact_fraction[-1]
    step_size = final_time.value / steps.value
    return exact_fraction, final_error, fraction, step_size, time


@app.cell
def _(exact_fraction, final_error, fraction, mo, rate, step_size):
    physical = bool(fraction.min() >= 0.0 and fraction.max() <= 1.0)
    stable_step = step_size * rate.value <= 1.0
    mo.callout(
        mo.md(
            fr"""
            ## Euler result

            - step size: **Δt = {step_size:.3f} min**
            - dimensionless Euler step $k\Delta t$: **{step_size * rate.value:.3f}**
            - Euler final fraction: **{fraction[-1]:.6f}**
            - exact final fraction: **{exact_fraction[-1]:.6f}**
            - final error: **{final_error:.3e}**
            - fraction remains in $[0,1]$: **{physical}**
            - simple monotonicity condition $k\Delta t\leq1$: **{stable_step}**

            Smaller steps improve accuracy for this smooth problem, but a solver
            tolerance is not a materials parameter. The rate law and its regime
            still determine whether the model is appropriate.
            """
        ),
        kind="success" if physical and abs(final_error) < 0.02 else "warn",
    )
    return


@app.cell
def _(exact_fraction, fraction, mo, time):
    width, height = 700, 320
    left, right, top, bottom = 58, 18, 24, 50
    x_pixels = left + time / max(time[-1], 1.0) * (width - left - right)
    y_pixels = top + (1.0 - fraction) * (height - top - bottom)
    exact_y = top + (1.0 - exact_fraction) * (height - top - bottom)
    euler_points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in zip(x_pixels, y_pixels)
    )
    exact_points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in zip(x_pixels, exact_y)
    )
    mo.Html(
        f"""
        <figure aria-label="Euler and exact transformation fraction over time">
          <svg viewBox="0 0 {width} {height}" role="img"
               style="max-width: 700px; width: 100%; height: auto;">
            <title>Euler trajectory compared with the exact solution</title>
            <polyline points="{exact_points}" fill="none" stroke="#d87700"
                      stroke-width="3" stroke-dasharray="7 4" />
            <polyline points="{euler_points}" fill="none" stroke="#007c41"
                      stroke-width="3" />
            <line x1="{left}" y1="{height - bottom}" x2="{width - right}"
                  y2="{height - bottom}" stroke="currentColor" />
            <line x1="{left}" y1="{top}" x2="{left}"
                  y2="{height - bottom}" stroke="currentColor" />
            <text x="{(left + width - right) / 2:.1f}" y="{height - 8}"
                  text-anchor="middle" font-size="14">time (min)</text>
            <text x="18" y="{(top + height - bottom) / 2:.1f}"
                  text-anchor="middle" font-size="14"
                  transform="rotate(-90 18 {(top + height - bottom) / 2:.1f})">transformed fraction</text>
            <text x="{width - 160}" y="22" font-size="13" fill="#007c41">Euler</text>
            <text x="{width - 160}" y="40" font-size="13" fill="#d87700">exact</text>
          </svg>
        </figure>
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Try this in the editor

            Change the update inside `euler_transformation`. Then reduce the
            number of steps until the fraction leaves its physical range or
            oscillates. The next lecture will separate accuracy from stability
            more systematically and compare established ODE integrators.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

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
    # Bisection: keep the equilibrium bracketed

    We continue the ceramic-surface heat-balance problem from L03. The
    residual $R(T_s)$ is continuous, and a sign change on $[a,b]$ traps at
    least one root. Bisection repeatedly halves that interval:

    1. evaluate the midpoint $m=(a+b)/2$;
    2. keep the half-interval whose endpoints have opposite signs;
    3. stop when the bracket width or residual is small enough.

    This is a deliberately transparent algorithm. It is not always the
    fastest method, but it is easy to reason about and robust when the
    initial bracket is valid.
    """)
    return


@app.cell
def _(mo):
    heat_flux = mo.ui.slider(
        start=2_000,
        stop=20_000,
        step=1_000,
        value=10_000,
        label="Incoming heat flux, q″ (W/m²)",
        show_value=True,
    )
    right_temperature = mo.ui.slider(
        start=500,
        stop=1_200,
        step=50,
        value=900,
        label="Right bracket endpoint, b (K)",
        show_value=True,
    )
    tolerance = mo.ui.dropdown(
        options={
            "1 K": 1.0,
            "0.1 K": 0.1,
            "0.01 K": 0.01,
        },
        value="0.1 K",
        label="Temperature tolerance",
    )
    mo.vstack([heat_flux, right_temperature, tolerance], align="start", gap=1)
    return heat_flux, right_temperature, tolerance


@app.cell
def _():
    STEFAN_BOLTZMANN = 5.670374419e-8
    CONVECTION_COEFFICIENT = 15.0
    AMBIENT_TEMPERATURE = 300.0
    EMISSIVITY = 0.80

    def heat_balance_residual(surface_temperature_K, heat_flux):
        # Keep the model visible while the solver remains in a separate cell.
        convection = CONVECTION_COEFFICIENT * (
            surface_temperature_K - AMBIENT_TEMPERATURE
        )
        radiation = EMISSIVITY * STEFAN_BOLTZMANN * (
            surface_temperature_K**4 - AMBIENT_TEMPERATURE**4
        )
        return convection + radiation - heat_flux

    return AMBIENT_TEMPERATURE, heat_balance_residual


@app.function
def bisection_root(residual_function, left, right, tolerance, max_iterations):
    # Given algorithm: edit the indented body and preserve the signature.
    left_value = residual_function(left)
    right_value = residual_function(right)
    history = []

    if left_value * right_value > 0.0:
        return float("nan"), history, "invalid bracket"

    for iteration in range(1, max_iterations + 1):
        midpoint = 0.5 * (left + right)
        midpoint_value = residual_function(midpoint)
        width = right - left
        history.append(
            (iteration, left, right, midpoint, midpoint_value, width)
        )

        if abs(midpoint_value) <= 1.0e-6 or width <= tolerance:
            return midpoint, history, "converged"

        if left_value * midpoint_value <= 0.0:
            right = midpoint
            right_value = midpoint_value
        else:
            left = midpoint
            left_value = midpoint_value

    return midpoint, history, "iteration limit"


@app.cell
def _(
    AMBIENT_TEMPERATURE,
    heat_balance_residual,
    heat_flux,
    np,
    right_temperature,
    tolerance,
):
    def residual_at_temperature(temperature):
        return heat_balance_residual(temperature, heat_flux.value)

    root, history, status = bisection_root(
        residual_at_temperature,
        AMBIENT_TEMPERATURE,
        right_temperature.value,
        float(tolerance.value),
        max_iterations=60,
    )
    root_residual = (
        residual_at_temperature(root) if np.isfinite(root) else float("nan")
    )
    return history, root, root_residual, status


@app.cell
def _(history, mo, root, root_residual, status):
    if history:
        last_iteration, left, right, midpoint, _, width = history[-1]
        detail = fr"""
        - iterations: **{last_iteration}**
        - final bracket: $[{left:.4f},\,{right:.4f}]\,\mathrm{{K}}$
        - final bracket width: **{width:.3e} K**
        """
    else:
        detail = "- No iteration history: inspect the endpoint signs and choose a valid bracket."

    if status == "converged":
        result = mo.callout(
            mo.md(
                fr"""
                ## Bisection result

                - status: **{status}**
                - surface temperature: **$T_s={root:.4f}\,\mathrm{{K}}$**
                - residual: **$R(T_s)={root_residual:.3e}\,\mathrm{{W/m^2}}$**
                {detail}

                The root is credible here because it remains inside a valid
                bracket and its residual is small. The tolerance is a numerical
                target, not a guarantee that the heat-transfer model is physically
                valid.
                """
            ),
            kind="success",
        )
    else:
        result = mo.callout(
            mo.md(f"**Bisection status:** `{status}`\n\n{detail}"),
            kind="warn",
        )
    result
    return


@app.cell
def _(history, mo, np):
    if history:
        iterations = np.array([row[0] for row in history], dtype=float)
        widths = np.array([row[5] for row in history], dtype=float)
        width_min = max(float(widths.min()), 1.0e-16)
        width_max = max(float(widths.max()), width_min * 10.0)
        width_range = np.log10(width_max) - np.log10(width_min)
        width_range = max(width_range, 1.0)
        width_points = " ".join(
            f"{40 + (iteration - 1) / max(iterations[-1] - 1, 1) * 630:.1f},"
            f"{25 + (np.log10(width_max) - np.log10(width)) / width_range * 240:.1f}"
            for iteration, width in zip(iterations, widths)
        )
        chart = mo.Html(
            f"""
            <figure aria-label="Bisection bracket width versus iteration">
              <svg viewBox="0 0 700 300" role="img"
                   style="max-width: 700px; width: 100%; height: auto;">
                <title>Bisection bracket width decreases with iteration</title>
                <polyline points="{width_points}" fill="none" stroke="#007c41"
                          stroke-width="3" />
                <line x1="40" y1="265" x2="670" y2="265" stroke="currentColor" />
                <line x1="40" y1="25" x2="40" y2="265" stroke="currentColor" />
                <text x="355" y="292" text-anchor="middle" font-size="14">iteration</text>
                <text x="14" y="145" text-anchor="middle" font-size="14"
                      transform="rotate(-90 14 145)">bracket width (log scale)</text>
              </svg>
            </figure>
            """
        )
    else:
        chart = mo.md("The convergence chart appears after a valid bracket is supplied.")
    chart
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Try this in the editor

            Change the body of `bisection_root` while keeping its inputs and
            returned values. Deliberately try an invalid bracket, an iteration
            limit, or a different stopping criterion, then restore the robust
            version. What evidence tells you that the algorithm—not merely the
            final printed number—behaved correctly?
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

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
    # Before solving: inspect the residual

    **Materials question.** A ceramic surface receives a uniform heat flux.
    What surface temperature makes the incoming and outgoing heat rates
    balance?

    We use a deliberately small steady-state model,

    $$
    q'' = h(T_s-T_\infty) + \varepsilon\sigma
    (T_s^4-T_\infty^4),
    $$

    and define the residual

    $$
    R(T_s) = h(T_s-T_\infty) + \varepsilon\sigma
    (T_s^4-T_\infty^4)-q''.
    $$

    The numerical problem is **not** “run a root finder.” First determine
    the physical domain and find an interval where the continuous residual
    changes sign. A root can only be trusted after we know what equation it
    represents and what range of temperature is admissible.
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
    surface = mo.ui.dropdown(
        options={
            "Polished metal · ε = 0.30": 0.30,
            "Ceramic coating · ε = 0.80": 0.80,
            "Black surface · ε = 0.95": 0.95,
        },
        value="Ceramic coating · ε = 0.80",
        label="Surface model",
    )
    right_temperature = mo.ui.slider(
        start=400,
        stop=1_200,
        step=50,
        value=900,
        label="Right end of trial interval, Tᵣ (K)",
        show_value=True,
    )
    mo.vstack([heat_flux, surface, right_temperature], align="start", gap=1)
    return heat_flux, right_temperature, surface


@app.cell
def _():
    STEFAN_BOLTZMANN = 5.670374419e-8
    CONVECTION_COEFFICIENT = 15.0
    AMBIENT_TEMPERATURE = 300.0

    def heat_balance_residual(
        surface_temperature_K,
        heat_flux,
        emissivity,
        ambient_temperature_K=AMBIENT_TEMPERATURE,
        convection_coefficient=CONVECTION_COEFFICIENT,
    ):
        # Keep this function signature. Try changing only the indented body.
        convection = convection_coefficient * (
            surface_temperature_K - ambient_temperature_K
        )
        radiation = emissivity * STEFAN_BOLTZMANN * (
            surface_temperature_K**4 - ambient_temperature_K**4
        )
        return convection + radiation - heat_flux

    return AMBIENT_TEMPERATURE, heat_balance_residual


@app.cell
def _(
    AMBIENT_TEMPERATURE,
    heat_balance_residual,
    heat_flux,
    np,
    right_temperature,
    surface,
):
    emissivity = surface.value
    left_temperature = AMBIENT_TEMPERATURE
    temperatures = np.linspace(left_temperature, right_temperature.value, 240)
    residuals = np.array(
        [
            heat_balance_residual(temperature, heat_flux.value, emissivity)
            for temperature in temperatures
        ]
    )
    left_residual = float(residuals[0])
    right_residual = float(residuals[-1])
    bracketed = left_residual * right_residual <= 0.0
    return bracketed, left_residual, residuals, right_residual, temperatures


@app.cell
def _(bracketed, left_residual, mo, right_residual):
    sign_kind = "success" if bracketed else "warn"
    mo.callout(
        mo.md(
            fr"""
            ## Bracket check

            - $R(T_\mathrm{{left}}) = $ **{left_residual:,.1f} W/m²**
            - $R(T_\mathrm{{right}}) = $ **{right_residual:,.1f} W/m²**
            - sign change on the trial interval: **{bracketed}**

            A sign change does not give the root yet. It gives evidence that a
            continuous residual has at least one root between the two endpoints.
            If the signs agree, change the trial interval before choosing a
            solver.
            """
        ),
        kind=sign_kind,
    )
    return


@app.cell
def _(mo, residuals, temperatures):
    def residual_plot(x_values, y_values):
        width, height = 700, 330
        left, right, top, bottom = 64, 18, 24, 50
        x_min, x_max = float(x_values.min()), float(x_values.max())
        y_min = min(float(y_values.min()), 0.0)
        y_max = max(float(y_values.max()), 0.0)
        y_span = max(y_max - y_min, 1.0)

        def x_pixel(value):
            return left + (value - x_min) / (x_max - x_min) * (
                width - left - right
            )

        def y_pixel(value):
            return top + (y_max - value) / y_span * (height - top - bottom)

        points = " ".join(
            f"{x_pixel(x):.1f},{y_pixel(y):.1f}"
            for x, y in zip(x_values, y_values)
        )
        zero_y = y_pixel(0.0)
        return mo.Html(
            f"""
            <figure aria-label="Heat-balance residual versus surface temperature">
              <svg viewBox="0 0 {width} {height}" role="img"
                   style="max-width: 700px; width: 100%; height: auto;">
                <title>Residual plot with a horizontal zero line</title>
                <line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}"
                      y2="{zero_y:.1f}" stroke="#9a7900" stroke-dasharray="6 4" />
                <polyline points="{points}" fill="none" stroke="#007c41"
                          stroke-width="3" />
                <line x1="{left}" y1="{top}" x2="{left}"
                      y2="{height - bottom}" stroke="currentColor" />
                <line x1="{left}" y1="{height - bottom}" x2="{width - right}"
                      y2="{height - bottom}" stroke="currentColor" />
                <text x="{(left + width - right) / 2:.1f}" y="{height - 10}"
                      text-anchor="middle" font-size="14">surface temperature, Tₛ (K)</text>
                <text x="18" y="{(top + height - bottom) / 2:.1f}"
                      text-anchor="middle" font-size="14"
                      transform="rotate(-90 18 {(top + height - bottom) / 2:.1f})">residual R(Tₛ) (W/m²)</text>
              </svg>
            </figure>
            """
        )

    residual_plot(temperatures, residuals)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Try this in the editor

            Open **Edit code** and find `heat_balance_residual`. Keep its name
            and inputs, but change the indented body. Then run that cell and
            inspect how the residual plot and endpoint signs respond.

            The programming task is intentionally local: write the equation in
            a supplied function. The engineering task is to decide whether the
            equation, units, physical domain, and bracket make sense.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

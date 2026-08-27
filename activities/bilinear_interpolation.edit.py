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
    # Interpolation is a local model

    A property table gives the thermal conductivity of compressed oxygen at
    selected temperatures and pressures. What should we report between the
    tabulated points?

    For a rectangular cell, bilinear interpolation takes a weighted average
    of the four corner values. The weights are determined by the relative
    position inside the cell. This is interpolation only while the query
    remains inside the data range; outside the table, the same formula is an
    extrapolation with much weaker evidence.
    """)
    return


@app.cell
def _(mo):
    temperature = mo.ui.slider(
        start=170,
        stop=210,
        step=1,
        value=190,
        label="Temperature, T (K)",
        show_value=True,
    )
    pressure = mo.ui.slider(
        start=0,
        stop=15,
        step=1,
        value=5,
        label="Pressure, P (atm)",
        show_value=True,
    )
    mo.vstack([temperature, pressure], align="start", gap=1)
    return pressure, temperature


@app.cell
def _(np):
    temperature_grid = np.array([180.0, 200.0])
    pressure_grid = np.array([1.0, 10.0])
    conductivity_grid = np.array(
        [[16.8e-3, 18.1e-3], [18.5e-3, 19.6e-3]]
    )

    def bilinear_interpolate(temperature, pressure):
        # Given function: keep the four-corner weighting visible.
        t0, t1 = temperature_grid
        p0, p1 = pressure_grid
        t_weight = (temperature - t0) / (t1 - t0)
        p_weight = (pressure - p0) / (p1 - p0)
        lower = (
            (1.0 - p_weight) * conductivity_grid[0, 0]
            + p_weight * conductivity_grid[0, 1]
        )
        upper = (
            (1.0 - p_weight) * conductivity_grid[1, 0]
            + p_weight * conductivity_grid[1, 1]
        )
        return (1.0 - t_weight) * lower + t_weight * upper

    return (
        bilinear_interpolate,
        conductivity_grid,
        pressure_grid,
        temperature_grid,
    )


@app.cell
def _(
    bilinear_interpolate,
    pressure,
    pressure_grid,
    temperature,
    temperature_grid,
):
    conductivity = bilinear_interpolate(temperature.value, pressure.value)
    inside_table = (
        temperature_grid[0] <= temperature.value <= temperature_grid[-1]
        and pressure_grid[0] <= pressure.value <= pressure_grid[-1]
    )
    t_weight = (temperature.value - temperature_grid[0]) / (
        temperature_grid[-1] - temperature_grid[0]
    )
    p_weight = (pressure.value - pressure_grid[0]) / (
        pressure_grid[-1] - pressure_grid[0]
    )
    return conductivity, inside_table, p_weight, t_weight


@app.cell
def _(
    conductivity,
    inside_table,
    mo,
    p_weight,
    pressure,
    t_weight,
    temperature,
):
    kind = "success" if inside_table else "warn"
    mo.callout(
        mo.md(
            fr"""
            ## Interpolated conductivity

            - query: $T={temperature.value}\,\mathrm{{K}}$, $P={pressure.value}\,\mathrm{{atm}}$
            - temperature weight: **{t_weight:.3f}**
            - pressure weight: **{p_weight:.3f}**
            - estimated conductivity: **{conductivity:.5f} W/(m·K)**
            - query inside table: **{inside_table}**

            {'This is interpolation: the query lies inside the measured rectangle.' if inside_table else 'This is extrapolation: the query lies outside the measured rectangle, so the same formula is a much weaker prediction.'}
            """
        ),
        kind=kind,
    )
    return


@app.cell
def _(conductivity_grid, mo, pressure, temperature):
    mo.md(
        fr"""
        ### Data used by the model

        | $T$ (K) / $P$ (atm) | 1 | 10 |
        |---:|---:|---:|
        | 180 | {conductivity_grid[0, 0]:.4f} | {conductivity_grid[0, 1]:.4f} |
        | 200 | {conductivity_grid[1, 0]:.4f} | {conductivity_grid[1, 1]:.4f} |

        At the default query $(T,P)=({temperature.value},\,{pressure.value})$,
        the code uses all four corners rather than fitting a high-degree global
        polynomial.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Try this in the editor

            Change `bilinear_interpolate` while keeping its inputs and four
            corner values. Test a corner, the cell midpoint, and an outside
            query. Which cases should reproduce a table entry exactly? Which
            cases should be reported as extrapolation rather than as a normal
            property prediction?
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

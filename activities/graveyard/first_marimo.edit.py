# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Your first marimo calculation

    **Materials question.** How much does a metal rod expand when its
    temperature changes?

    We will use the small-strain model

    $$
    \Delta L = \alpha L_0 \Delta T,
    $$

    where $L_0$ is the initial length, $\alpha$ is the linear thermal
    expansion coefficient, and $\Delta T$ is the temperature change.

    You do **not** need to design a software package for this exercise.
    The function signature is supplied: write or change the expression
    inside the function, run the cell, and let marimo update the result.
    """)
    return


@app.cell
def _(mo):
    length0_mm = mo.ui.slider(
        start=50,
        stop=500,
        step=10,
        value=100,
        label="Initial length, L₀ (mm)",
        show_value=True,
    )
    alpha_micro = mo.ui.slider(
        start=5,
        stop=30,
        step=1,
        value=23,
        label="Expansion coefficient, α (µm/m/K)",
        show_value=True,
    )
    temperature_change_K = mo.ui.slider(
        start=-100,
        stop=400,
        step=10,
        value=100,
        label="Temperature change, ΔT (K)",
        show_value=True,
    )
    mo.vstack(
        [length0_mm, alpha_micro, temperature_change_K],
        align="start",
        gap=1,
    )
    return alpha_micro, length0_mm, temperature_change_K


@app.function
def thermal_expansion(length0_mm, alpha_per_K, delta_temperature_K):
    # Keep this function signature. Try changing only the indented body.
    # Units: mm × 1/K × K = mm.
    return length0_mm * alpha_per_K * delta_temperature_K


@app.cell
def _(alpha_micro, length0_mm, temperature_change_K):
    alpha_per_K = alpha_micro.value * 1.0e-6
    elongation_mm = thermal_expansion(
        length0_mm.value,
        alpha_per_K,
        temperature_change_K.value,
    )
    final_length_mm = length0_mm.value + elongation_mm
    zero_change_check = thermal_expansion(
        length0_mm.value,
        alpha_per_K,
        0.0,
    )
    return alpha_per_K, elongation_mm, final_length_mm, zero_change_check


@app.cell
def _(
    alpha_per_K,
    elongation_mm,
    final_length_mm,
    length0_mm,
    mo,
    temperature_change_K,
    zero_change_check,
):
    mo.callout(
        mo.md(
            fr"""
            ## Result

            For $L_0 = {length0_mm.value:.0f}\,\mathrm{{mm}}$,
            $\alpha = {alpha_per_K:.2e}\,\mathrm{{K^{{-1}}}}$, and
            $\Delta T = {temperature_change_K.value:.0f}\,\mathrm{{K}}$:

            - predicted change: **$\Delta L = {elongation_mm:.4f}\,\mathrm{{mm}}$**
            - final length: **$L = {final_length_mm:.4f}\,\mathrm{{mm}}$**
            - limiting-case check ($\Delta T=0$):
              $\Delta L = {zero_change_check:.1e}\,\mathrm{{mm}}$

            The last line is a tiny verification check: no temperature change
            should produce no thermal expansion in this model.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Try this in the editor

            1. Open **Edit code** and find `thermal_expansion`.
            2. Keep the function name and three inputs unchanged.
            3. Change the expression after `return`, or add an intermediate
               variable inside the function.
            4. Run that cell. The result cell depends on it and should update.

            This is enough Python for today: inputs, an indented function body,
            a returned value, and a downstream calculation. Function-wrapping and
            more formal functional-programming patterns can wait for the seminar.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

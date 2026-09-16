# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import math
    import marimo as mo

    return math, mo


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## One CO₂ volume update

    **Predict:** will the next volume be larger or smaller?
    Enter the current molar volume, record the result, then copy the new volume
    into the input yourself. Each calculation performs one update.
    """)
    return


@app.cell(hide_code=False)
def parameters():
    a = 3.592          # L² bar / mol²
    b = 0.04267        # L / mol
    R = 0.08314462618  # L bar / (mol K)
    T = 280.0         # K
    P = 50.0          # bar
    volume_tolerance = 0.001  # L / mol
    pressure_tolerance = 0.2  # bar

    def F(v):
        return R * T / (v - b) - a / v**2 - P

    return F, P, R, T, a, b, pressure_tolerance, volume_tolerance


@app.cell(hide_code=False)
def update_formula(P, R, T, a, b):
    def g(v):
        # Replace this expression to try another rearrangement.
        return b + R * T / (P + a / v**2)

    return (g,)


@app.cell(hide_code=True)
def volume_input(P, R, T, mo):
    current_volume = mo.ui.number(
        value=R * T / P, step=0.0001, label="Current v (L/mol)"
    )
    current_volume
    return (current_volume,)


@app.cell(hide_code=True)
def single_update(F, b, g, math):
    def evaluate_update(v):
        if v is None or not math.isfinite(v) or v <= b:
            raise ValueError(f"Enter a finite volume greater than b = {b} L/mol.")
        next_v = g(v)
        if not math.isfinite(next_v) or next_v <= b:
            raise ValueError(f"The proposed volume must be finite and greater than b = {b} L/mol.")
        residual = F(next_v)
        if not math.isfinite(residual):
            raise ValueError("The pressure residual is not finite at the proposed volume.")
        return next_v, abs(next_v - v), residual

    return (evaluate_update,)


@app.cell(hide_code=True)
def result(current_volume, evaluate_update, mo, pressure_tolerance, volume_tolerance):
    try:
        next_volume, step, residual = evaluate_update(current_volume.value)
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        calculation_output = mo.callout(
            mo.md(f"**No valid update.** {error}\n\nCheck the input and the domain of `g(v)`."),
            kind="warn",
        )
    else:
        step_passes = step < volume_tolerance
        residual_passes = abs(residual) < pressure_tolerance
        accepted = step_passes and residual_passes
        calculation_output = mo.vstack([
            mo.md(
                f"**New volume:** {next_volume:.10g} L/mol\n\n"
                "| Check | Value | Result |\n"
                "|---|---:|---|\n"
                f"| Step | {step:.4g} L/mol | {'Pass' if step_passes else 'Fail'}: < {volume_tolerance:g} L/mol |\n"
                f"| Pressure residual F(new volume) | {residual:+.4g} bar | {'Pass' if residual_passes else 'Fail'}: magnitude < {pressure_tolerance:g} bar |"
            ),
            mo.callout(
                mo.md("**Both checks pass:** accept under the seminar stopping rule."
                      if accepted else "**Not converged under the seminar stopping rule.** Copy the new volume into the input to take the next step."),
                kind="success" if accepted else "warn",
            ),
        ])
    calculation_output
    return (calculation_output,)


if __name__ == "__main__":
    app.run()

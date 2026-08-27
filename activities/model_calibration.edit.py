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
    # Calibration, validation, and identifiability

    A small residual on the data used to fit a model is not the same as
    predictive validation. We use a synthetic diffusion data set with a
    small deterministic measurement/model discrepancy and compare two
    candidate forms:

    $$
    D=D_0\exp\!\left(-\frac{E_a}{RT}\right),
    \qquad
    D=C T^m.
    $$

    For each run, one temperature is held out. The question is not which
    curve looks nicest on the training points; it is whether the data and
    temperature range identify a defensible prediction.
    """)
    return


@app.cell
def _(mo):
    holdout = mo.ui.dropdown(
        options={
            "Hold out 900 K": 0,
            "Hold out 1100 K": 2,
            "Hold out 1300 K": 4,
            "Hold out 1400 K": 5,
        },
        value="Hold out 1300 K",
        label="Validation point",
    )
    holdout
    return (holdout,)


@app.cell
def _(np):
    gas_constant = 8.314
    temperature = np.array([900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0])
    ideal_diffusivity = 2.0e-4 * np.exp(-180_000.0 / (gas_constant * temperature))
    discrepancy = np.array([1.02, 0.97, 1.04, 0.96, 1.03, 0.99])
    diffusivity = ideal_diffusivity * discrepancy

    def fit_linear_model(x, y):
        # Given function: fit transformed variables using a design matrix.
        design = np.column_stack([np.ones_like(x), x])
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        return coefficients

    return diffusivity, fit_linear_model, temperature


@app.cell
def _(diffusivity, fit_linear_model, holdout, np, temperature):
    validation_index = int(holdout.value)
    training_mask = np.ones(len(temperature), dtype=bool)
    training_mask[validation_index] = False
    train_temperature = temperature[training_mask]
    train_diffusivity = diffusivity[training_mask]
    train_log_diffusivity = np.log(train_diffusivity)

    arrhenius_coefficients = fit_linear_model(
        1.0 / train_temperature, train_log_diffusivity
    )
    power_coefficients = fit_linear_model(
        np.log(train_temperature), train_log_diffusivity
    )

    def arrhenius_prediction(temp):
        return np.exp(
            arrhenius_coefficients[0]
            + arrhenius_coefficients[1] / temp
        )

    def power_prediction(temp):
        return np.exp(
            power_coefficients[0]
            + power_coefficients[1] * np.log(temp)
        )

    predictions = np.array(
        [arrhenius_prediction(temperature[validation_index]), power_prediction(temperature[validation_index])]
    )
    observed = diffusivity[validation_index]
    relative_errors = (predictions - observed) / observed
    train_arrhenius_error = np.sqrt(
        np.mean(
            (
                np.log(train_diffusivity)
                - np.log(arrhenius_prediction(train_temperature))
            )
            ** 2
        )
    )
    train_power_error = np.sqrt(
        np.mean(
            (
                np.log(train_diffusivity)
                - np.log(power_prediction(train_temperature))
            )
            ** 2
        )
    )
    return (
        observed,
        predictions,
        relative_errors,
        train_arrhenius_error,
        train_power_error,
        validation_index,
    )


@app.cell
def _(
    mo,
    observed,
    predictions,
    relative_errors,
    temperature,
    train_arrhenius_error,
    train_power_error,
    validation_index,
):
    rows = "\n".join(
        f"| Arrhenius | {predictions[0]:.3e} | {relative_errors[0]:.2%} | {train_arrhenius_error:.3e} |"
        f"\n| Power law | {predictions[1]:.3e} | {relative_errors[1]:.2%} | {train_power_error:.3e} |"
    )
    mo.callout(
        mo.md(
            f"""
            ## Held-out prediction

            Validation temperature: **{temperature[validation_index]:.0f} K**  
            Observed diffusivity: **{observed:.3e} m²/s**

            | Model | held-out prediction (m²/s) | validation error | training RMS log error |
            |---|---:|---:|---:|
            {rows}

            A model with the smallest training error is not automatically the
            best model outside the calibration points. With only a few data
            points, this validation comparison is indicative rather than final.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Interpretation prompt

            What additional data would make the mechanism more identifiable?
            Consider a wider temperature range, replicated measurements,
            uncertainty estimates, or a physically justified competing model.
            Do not report more significant digits than the data can support.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

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
    # Arrhenius regression: what can the data support?

    Diffusion coefficients often follow an Arrhenius-like relation,

    $$
    D(T)=D_0\exp\!\left(-\frac{E_a}{RT}\right).
    $$

    Taking logarithms gives a linear model,

    $$
    \ln D = \ln D_0 - \frac{E_a}{R}\frac1T.
    $$

    We fit the transformed data, then return to the original units. The
    transformation is a modeling choice, not a way to make uncertainty
    disappear. Residuals should be inspected in both transformed and
    physical spaces.
    """)
    return


@app.cell
def _(mo):
    validation = mo.ui.dropdown(
        options={
            "Fit all measurements": "all",
            "Hold out the highest-temperature point": "holdout",
        },
        value="Fit all measurements",
        label="Calibration / validation mode",
    )
    validation
    return (validation,)


@app.cell
def _(np):
    temperature_K = np.array([1100.0, 1150.0, 1200.0, 1250.0])
    diffusivity_m2_s = np.array(
        [1.60e-18, 6.41e-18, 2.10e-17, 6.70e-17]
    )
    gas_constant = 8.314

    def fit_arrhenius(temperature, diffusivity):
        # Given function: edit the design matrix or return calculation.
        inverse_temperature = 1.0 / temperature
        log_diffusivity = np.log(diffusivity)
        design = np.column_stack(
            [np.ones_like(inverse_temperature), inverse_temperature]
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design, log_diffusivity, rcond=None
        )
        log_prefactor, inverse_temperature_slope = coefficients
        return np.exp(log_prefactor), -inverse_temperature_slope * gas_constant

    return diffusivity_m2_s, fit_arrhenius, temperature_K


@app.cell
def _(diffusivity_m2_s, fit_arrhenius, np, temperature_K, validation):
    if validation.value == "holdout":
        training_mask = np.arange(len(temperature_K)) < len(temperature_K) - 1
    else:
        training_mask = np.ones(len(temperature_K), dtype=bool)

    prefactor, activation_energy = fit_arrhenius(
        temperature_K[training_mask], diffusivity_m2_s[training_mask]
    )
    predicted_diffusivity = prefactor * np.exp(
        -activation_energy / (8.314 * temperature_K)
    )
    log_residuals = np.log(diffusivity_m2_s) - np.log(predicted_diffusivity)
    relative_residuals = (
        diffusivity_m2_s - predicted_diffusivity
    ) / diffusivity_m2_s
    return (
        activation_energy,
        log_residuals,
        predicted_diffusivity,
        prefactor,
        relative_residuals,
        training_mask,
    )


@app.cell
def _(
    activation_energy,
    diffusivity_m2_s,
    log_residuals,
    mo,
    np,
    predicted_diffusivity,
    prefactor,
    relative_residuals,
    temperature_K,
    training_mask,
    validation,
):
    rows = "\n".join(
        f"| {temperature:.0f} | {observed:.3e} | {predicted:.3e} | {relative:.2%} | {'fit' if used else 'holdout'} |"
        for temperature, observed, predicted, relative, used in zip(
            temperature_K,
            diffusivity_m2_s,
            predicted_diffusivity,
            relative_residuals,
            training_mask,
        )
    )
    holdout_text = ""
    if validation.value == "holdout":
        holdout_error = abs(relative_residuals[-1])
        holdout_text = f"The held-out point has a relative prediction error of **{holdout_error:.2%}**."
    mo.callout(
        mo.md(
            f"""
            ## Arrhenius fit

            - $D_0$: **{prefactor:.3e} m²/s**
            - $E_a$: **{activation_energy / 1_000:.2f} kJ/mol**
            - RMS log-space residual: **{np.sqrt(np.mean(log_residuals**2)):.3e}**

            | $T$ (K) | measured $D$ (m²/s) | predicted $D$ (m²/s) | relative residual | role |
            |---:|---:|---:|---:|---|
            {rows}

            {holdout_text}

            A good fit is not automatically a good predictive model. Check the
            residual pattern, the temperature range, units, and whether an
            Arrhenius mechanism is justified.
            """
        ),
        kind="success" if validation.value == "all" else "info",
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Try this in the editor

            Change the body of `fit_arrhenius` while keeping its inputs and
            returned quantities. Compare the transformed fit with the physical
            diffusivity errors. The linear algebra is a tool for calibration;
            the materials judgment is deciding whether the fitted parameters
            deserve interpretation.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

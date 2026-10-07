# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A3 Q3 · Predicting energy from nutrition labels

    Can macronutrient masses help predict the energy reported on a nutrition label?
    We will estimate the coefficients in the model

    $$E=c_f f+c_p p+c_c c,$$

    where $f$, $p$, and $c$ are the masses of fat, protein, and carbohydrate in
    grams, and $E$ is energy in kilocalories. The labels in the table provide the
    data for a least-squares estimate of $c_f$, $c_p$, and $c_c$.

    ## How to use the table

    - Use supplied labels 1–3 for the initial model. The fourth supplied
      example checks that prediction and is included in the later fit.

    - Enter **3–5 additional labels** in the blank rows. The highest-energy
      additional label is held out from the least-squares fit; all four
      supplied labels and the remaining additions are used for fitting.

    - Select **Calculate** to submit the table.
    """)
    return


@app.cell(hide_code=True)
def _():
    label_rows = [
        {"Fat (g)": 30.0, "Protein (g)": 3.6, "Carbohydrate (g)": 51.6, "Energy (kcal)": 490},
        {"Fat (g)": 0.0, "Protein (g)": 8.0, "Carbohydrate (g)": 15.0, "Energy (kcal)": 90},
        {"Fat (g)": 0.5, "Protein (g)": 1.0, "Carbohydrate (g)": 3.0, "Energy (kcal)": 20},
        {"Fat (g)": 0.5, "Protein (g)": 0.0, "Carbohydrate (g)": 2.0, "Energy (kcal)": 15},
    ]
    label_rows += [
        {field: "" for field in ("Fat (g)", "Protein (g)", "Carbohydrate (g)", "Energy (kcal)")}
        for _ in range(5)
    ]
    return (label_rows,)


@app.cell
def _(label_rows, mo, np):
    _data = np.array([[float(_row[_field]) for _field in
                      ("Fat (g)", "Protein (g)", "Carbohydrate (g)", "Energy (kcal)")]
                     for _row in label_rows[:4]])
    _coefficients = np.linalg.solve(_data[:3, :3], _data[:3, 3])
    _prediction = float(_data[3, :3] @ _coefficients)
    _error = abs(_prediction - _data[3, 3])
    mo.md(f"""
    ## Q3.1–3.2 · Initial three-label model

    Coefficients (fat, protein, carbohydrate): **{_coefficients[0]:.3f},
    {_coefficients[1]:.3f}, {_coefficients[2]:.3f} kcal/g**.

    For the fourth supplied label (15 kcal), the prediction is
    **{_prediction:.3f} kcal**, with absolute error **{_error:.3f} kcal**
    and relative error **{_error / _data[3, 3]:.1%}**.
    This label joins the later least-squares fit.
    """)
    return


@app.cell(hide_code=True)
def _(label_rows, mo):
    label_input = mo.ui.data_editor(
        data=label_rows,
        label="Nutrition label values",
    ).form(submit_button_label="Calculate")
    label_input
    return (label_input,)


@app.cell
def _(label_input, mo, np):
    mo.stop(
        label_input.value is None,
        mo.callout(
            mo.md("Enter the label values and select **Calculate** to submit the table."),
            kind="info",
        ),
    )
    _submitted = list(label_input.value)
    _fields = ("Fat (g)", "Protein (g)", "Carbohydrate (g)", "Energy (kcal)")
    _used_rows = [
        _row for _row in _submitted
        if any(_row.get(_field) is not None and str(_row.get(_field)).strip() for _field in _fields)
    ]
    mo.stop(
        len(_used_rows) < 7,
        mo.callout(
            mo.md(
                "Add at least three labels of your own. The four supplied labels and "
                "your additions provide at least seven examples; the highest-energy "
                "additional label is held out, leaving six fitting rows."
            ),
            kind="warn",
        ),
    )
    try:
        all_data = np.asarray(
            [[float(_row[_field]) for _field in _fields] for _row in _used_rows],
            dtype=float,
        )
        _valid_data = np.isfinite(all_data).all() and (all_data >= 0).all()
    except (KeyError, TypeError, ValueError):
        all_data = np.empty((0, 4))
        _valid_data = False
    mo.stop(
        not _valid_data,
        mo.callout(
            mo.md("Complete every used row with finite, nonnegative values in all four columns."),
            kind="warn",
        ),
    )
    held_out_index = 4 + int(np.argmax(all_data[4:, 3]))
    held_out_nutrients = all_data[held_out_index, :3]
    held_out_energy = float(all_data[held_out_index, 3])
    mo.stop(
        held_out_energy <= 0,
        mo.callout("Enter an additional label with positive energy to compute a relative error.", kind="warn"),
    )
    fit_data = np.delete(all_data, held_out_index, axis=0)
    return fit_data, held_out_nutrients, held_out_energy


@app.cell
def least_squares_calculation(fit_data, held_out_energy, held_out_nutrients, np):
    # Each row of A_fit contains fat, protein, and carbohydrate in that order.
    A_fit = fit_data[:, :3]
    E_fit = fit_data[:, 3]

    # Find c_f, c_p, and c_c that minimize ||A_fit @ coefficients - E_fit||².
    coefficients, _, _, _ = np.linalg.lstsq(A_fit, E_fit, rcond=None)

    fit_predictions = A_fit @ coefficients
    fit_residuals = fit_predictions - E_fit
    condition_number = float(np.linalg.cond(A_fit))

    # The highest-energy additional example stays out of the fit.
    held_out_prediction = float(held_out_nutrients @ coefficients)
    held_out_error = abs(held_out_prediction - held_out_energy)
    held_out_relative_error = held_out_error / held_out_energy
    return (
        A_fit,
        E_fit,
        coefficients,
        condition_number,
        fit_predictions,
        fit_residuals,
        held_out_error,
        held_out_prediction,
        held_out_relative_error,
        held_out_nutrients,
        held_out_energy,
    )


@app.cell
def least_squares_results(
    A_fit,
    E_fit,
    coefficients,
    condition_number,
    fit_predictions,
    fit_residuals,
    held_out_error,
    held_out_prediction,
    held_out_relative_error,
    held_out_nutrients,
    held_out_energy,
    mo,
):
    mo.vstack([
        mo.md(
            f"""## How is the least-squares fit?

    - **Total data input:** N = {len(E_fit) + 1} labels ({len(E_fit)} fitting labels and 1 held-out label).
    - **Condition number of the fitting matrix $A$:** {condition_number:.3g}."""
        ),
        mo.ui.table([
            {"Coefficient": _symbol, "Estimate (kcal/g)": round(float(_value), 4)}
            for _symbol, _value in zip(("c_f", "c_p", "c_c"), coefficients)
        ], selection=None),
        mo.md(
            f"""## How accurate is the held-out prediction?

    The highest-energy additional example is held out. Its entered nutrient masses and printed energy appear in the final row of the table below.

    - **Prediction:** {held_out_prediction:.1f} kcal.
    - **Absolute error:** {held_out_error:.1f} kcal.
    - **Relative error:** {held_out_relative_error:.1%} (absolute error divided by reported energy)."""
        ),
        mo.md("## Fitting examples and final held-out example"),
        mo.ui.table([
            {
                "Use": "Fit",
                "Fat (g)": float(_row[0]),
                "Protein (g)": float(_row[1]),
                "Carbohydrate (g)": float(_row[2]),
                "Printed energy (kcal)": float(E_fit[_i]),
                "Predicted energy (kcal)": round(float(fit_predictions[_i]), 2),
                "Prediction error (predicted − printed, kcal)": round(float(fit_residuals[_i]), 2),
            }
            for _i, _row in enumerate(A_fit)
        ] + [{
            "Use": "Held out",
            "Fat (g)": float(held_out_nutrients[0]),
            "Protein (g)": float(held_out_nutrients[1]),
            "Carbohydrate (g)": float(held_out_nutrients[2]),
            "Printed energy (kcal)": held_out_energy,
            "Predicted energy (kcal)": round(held_out_prediction, 2),
            "Prediction error (predicted − printed, kcal)": round(held_out_prediction - held_out_energy, 2),
        }], selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()

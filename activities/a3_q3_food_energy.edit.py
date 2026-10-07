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
    grams, and $E$ is energy in kilocalories. Labels 1–3 in the assignment
    figure give an exact three-label solution. The fourth label in the figure
    is the held-out label (23 g fat, 3 g protein, 15 g carbohydrate, 270 kcal).
    It never enters a fit, so both the three-label model and the
    least-squares model are tested on the same food.
    """)
    return


@app.cell(hide_code=True)
def _():
    label_rows = [
        # Labels 1–3 from the assignment figure.
        {"Fat (g)": 30.0, "Protein (g)": 3.6, "Carbohydrate (g)": 51.6, "Energy (kcal)": 490.0},
        {"Fat (g)": 0.0, "Protein (g)": 8.0, "Carbohydrate (g)": 15.0, "Energy (kcal)": 90.0},
        {"Fat (g)": 0.5, "Protein (g)": 1.0, "Carbohydrate (g)": 3.0, "Energy (kcal)": 20.0},
    ]
    # The held-out label from the assignment figure.
    held_out_label = {"Fat (g)": 23.0, "Protein (g)": 3.0, "Carbohydrate (g)": 15.0, "Energy (kcal)": 270}
    return held_out_label, label_rows


@app.cell
def _(held_out_label, label_rows, np):
    # The code below shows how to solve the coefficients from labels 1–3 of the
    # assignment figure and test them on the held-out label. See the comments for details.
    _fields = ("Fat (g)", "Protein (g)", "Carbohydrate (g)", "Energy (kcal)")

    # These are labels 1–3 from the assignment figure, one row per label
    _data = np.array([[float(_row[_field]) for _field in _fields]
                      for _row in label_rows[:3]])

    # extract the matrix A and vector E from the data
    A_first_three = _data[:, :3]  # rows 0-2, cols 0-2 (fat, protein, carbohydrate)
    E_first_three = _data[:, 3]  # rows 0-2, col 3 (energy)

    # nutrient masses and printed energy of the held-out label
    a_held_out = np.array([float(held_out_label[_field]) for _field in _fields[:3]])
    e_held_out = float(held_out_label["Energy (kcal)"])

    # coefficients solved using np.linalg.solve for AC = E
    C_first_three = np.linalg.solve(A_first_three, E_first_three)

    # prediction on the held-out label
    e_prediction_first_three = a_held_out @ C_first_three

    # abs and rel error of that prediction on the held-out label
    abs_error_first_three = np.abs(e_prediction_first_three - e_held_out)
    rel_error_first_three = abs_error_first_three / e_held_out
    C_first_three, abs_error_first_three, rel_error_first_three
    return (
        A_first_three,
        C_first_three,
        E_first_three,
        a_held_out,
        abs_error_first_three,
        e_held_out,
        e_prediction_first_three,
        rel_error_first_three,
    )


@app.cell(hide_code=True)
def _(
    A_first_three,
    C_first_three,
    E_first_three,
    abs_error_first_three,
    e_held_out,
    e_prediction_first_three,
    mo,
    rel_error_first_three,
):
    _A_entries = r" \\ ".join(" & ".join(f"{_value:g}" for _value in _row) for _row in A_first_three)
    _E_entries = r" \\ ".join(f"{_value:g}" for _value in E_first_three)
    mo.md(rf"""
    ## Reference answer Q3.1–3.2

    - **Worked-out $\mathbf A$** (`A_first_three`): $\begin{{bmatrix}}{_A_entries}\end{{bmatrix}}$ g
    - **Worked-out $\mathbf E$** (`E_first_three`): $\begin{{bmatrix}}{_E_entries}\end{{bmatrix}}$ kcal
    - **Solved coefficients** (`C_first_three`): $c_f={C_first_three[0]:.3f}$, $c_p={C_first_three[1]:.3f}$, $c_c={C_first_three[2]:.3f}$ kcal/g
    - **On the held-out label** ({e_held_out:g} kcal printed):
        - Predicted energy: {e_prediction_first_three:.1f} kcal
        - Absolute error: {abs_error_first_three:.1f} kcal, relative error: {rel_error_first_three:.1%}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q3.4 · Least-squares table

    - The table starts with labels 1–3 from the assignment figure.

    - Add **3–5 of your own labels**: click the **+** row at the bottom of the
      table to append a row, then enter its fat, protein, carbohydrate, and
      energy. Every row in the table is used in the least-squares fit.

    - The held-out label stays outside the table. The least-squares
      coefficients predict its energy, so the error can be compared directly
      with Q3.2.

    - Select **Calculate** to submit the table.
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


@app.cell(hide_code=True)
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
        len(_used_rows) < 6,
        mo.callout(
            mo.md(
                "Add at least three labels of your own with the **+** row at the "
                "bottom of the table. Together with labels 1–3, they give at least "
                "six fitting rows."
            ),
            kind="warn",
        ),
    )
    try:
        total_data = np.asarray(
            [[float(_row[_field]) for _field in _fields] for _row in _used_rows],
            dtype=float,
        )
        _valid_data = np.isfinite(total_data).all() and (total_data >= 0).all()
    except (KeyError, TypeError, ValueError):
        total_data = np.empty((0, 4))
        _valid_data = False
    mo.stop(
        not _valid_data,
        mo.callout(
            mo.md("Complete every used row with finite, nonnegative values in all four columns."),
            kind="warn",
        ),
    )
    return (total_data,)


@app.cell
def least_squares_calculation(a_held_out, e_held_out, total_data, np):
    # Each row of A_total contains fat, protein, and carbohydrate in that order.
    A_total = total_data[:, :3]
    E_total = total_data[:, 3]

    # Find c_f, c_p, and c_c that minimize ||A_total @ C_fit - E_total||².
    C_fit, _, _, _ = np.linalg.lstsq(A_total, E_total, rcond=None)

    fit_predictions = A_total @ C_fit
    fit_residuals = fit_predictions - E_total
    condition_number = float(np.linalg.cond(A_total))

    # The held-out label from Q3.2 stays out of the fit and tests the new coefficients.
    e_prediction_fit = float(a_held_out @ C_fit)
    abs_error_fit = abs(e_prediction_fit - e_held_out)
    rel_error_fit = abs_error_fit / e_held_out
    return (
        A_total,
        C_fit,
        E_total,
        abs_error_fit,
        condition_number,
        e_prediction_fit,
        fit_predictions,
        fit_residuals,
        rel_error_fit,
    )


@app.cell(hide_code=True)
def least_squares_results(
    A_total,
    C_fit,
    E_total,
    a_held_out,
    abs_error_first_three,
    abs_error_fit,
    condition_number,
    e_held_out,
    e_prediction_first_three,
    e_prediction_fit,
    fit_predictions,
    fit_residuals,
    mo,
    rel_error_first_three,
    rel_error_fit,
):
    mo.vstack([
        mo.md(
            f"""## Least-squares coefficients

    - **Fitting rows:** {len(E_total)} labels. The held-out label is not among them.
    - **Condition number of `A_total`:** {condition_number:.3g}."""
        ),
        mo.ui.table([
            {"Coefficient": _symbol, "Estimate (kcal/g)": round(float(_value), 4)}
            for _symbol, _value in zip(("c_f", "c_p", "c_c"), C_fit)
        ], selection=None),
        mo.md(
            f"""## Held-out prediction: three labels versus least squares

    Both models predict the same held-out label, printed as {e_held_out:g} kcal."""
        ),
        mo.ui.table([
            {
                "Model": "Three labels (Q3.2)",
                "Predicted energy (kcal)": round(float(e_prediction_first_three), 2),
                "Absolute error (kcal)": round(float(abs_error_first_three), 2),
                "Relative error (%)": round(100 * float(rel_error_first_three), 2),
            },
            {
                "Model": f"Least squares, {len(E_total)} labels (Q3.4)",
                "Predicted energy (kcal)": round(e_prediction_fit, 2),
                "Absolute error (kcal)": round(abs_error_fit, 2),
                "Relative error (%)": round(100 * rel_error_fit, 2),
            },
        ], selection=None),
        mo.md("## Fitting rows and the held-out label"),
        mo.ui.table([
            {
                "Use": "Fit",
                "Fat (g)": float(_row[0]),
                "Protein (g)": float(_row[1]),
                "Carbohydrate (g)": float(_row[2]),
                "Printed energy (kcal)": float(E_total[_i]),
                "Predicted energy (kcal)": round(float(fit_predictions[_i]), 2),
                "Prediction error (predicted − printed, kcal)": round(float(fit_residuals[_i]), 2),
            }
            for _i, _row in enumerate(A_total)
        ] + [{
            "Use": "Held out",
            "Fat (g)": float(a_held_out[0]),
            "Protein (g)": float(a_held_out[1]),
            "Carbohydrate (g)": float(a_held_out[2]),
            "Printed energy (kcal)": e_held_out,
            "Predicted energy (kcal)": round(e_prediction_fit, 2),
            "Prediction error (predicted − printed, kcal)": round(e_prediction_fit - e_held_out, 2),
        }], selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()

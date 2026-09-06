# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # A1 playground: when does an increment disappear?

    Floating-point numbers have finite spacing. This playground adds
    $10^{-n}$ to 1.0 using three NumPy precisions. Change the precision
    and exponent, then compare the requested increment with the spacing
    between representable values near 1.0.
    """)
    return


@app.cell
def _(mo):
    precision = mo.ui.dropdown(
        options=["float16", "float32", "float64"],
        value="float32",
        label="Floating-point type",
    )
    exponent = mo.ui.slider(
        start=1,
        stop=20,
        step=1,
        value=8,
        label="Exponent n in the increment 10⁻ⁿ",
        show_value=True,
    )
    mo.hstack([precision, exponent], widths="equal", gap=2)
    return exponent, precision


@app.cell
def _(exponent, mo, np, precision):
    def _evaluate_increment():
        dtype = getattr(np, precision.value)
        one = dtype(1.0)
        requested_increment = 10.0 ** (-exponent.value)
        stored_increment = dtype(requested_increment)
        updated = dtype(one + stored_increment)
        spacing = np.spacing(one)
        changed = bool(updated != one)

        return mo.callout(
            mo.md(
                f"""
                ## Stored result

                - type: `{precision.value}`
                - requested increment: `{requested_increment:.3e}`
                - spacing near 1.0: `{float(spacing):.3e}`
                - stored `1.0 + increment`: `{float(updated):.20g}`
                - did the stored value change? **{changed}**

                Compare the increment with the spacing. The transition is not
                a fixed absolute error for every magnitude; move the baseline
                away from 1.0 in the editor to test that statement.
                """
            ),
            kind="success" if changed else "warn",
        )

    _evaluate_increment()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Code pattern to reuse

    Open **Edit code** to inspect the calculation. For the assignment,
    also use `np.finfo(np.float32).eps` and `np.finfo(np.float64).eps`,
    record the actual outputs, and explain them in your own words.
    """)
    return


if __name__ == "__main__":
    app.run()

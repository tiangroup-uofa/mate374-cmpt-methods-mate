# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from decimal import Decimal
    from fractions import Fraction
    return Decimal, Fraction, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### What number did we actually store?

    Start with **0.75**, then try **0.3**. Compare the exact stored values:
    a short decimal display can hide the difference between the two formats.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    number = mo.ui.dropdown(
        options=["0.75", "0.3", "0.1", "-13", "10240", "1e20", "1e-30"],
        value="0.75", label="Decimal input",
    )
    number
    return (number,)


@app.cell(hide_code=True)
def _(Decimal, Fraction, mo, np, number):
    def describe_stored(text, dtype):
        stored = dtype(text)
        # Every float32 value is exactly representable as a Python float.
        numerator, denominator = float(stored).as_integer_ratio()
        bits = bin(abs(numerator))[2:]
        k = denominator.bit_length() - 1
        exponent = len(bits) - 1 - k
        fraction_bits = bits[1:].rstrip("0") or "0"
        sign = "-" if numerator < 0 else ""
        exact = Fraction(numerator, denominator) == Fraction(text)
        return mo.md(f"""
        ### {dtype.__name__}

        Short display: `{str(stored)}`

        **Exact match to input?** {"Yes" if exact else "No"}

        **1. Stored ratio**

        `{numerator} / {denominator}`

        = `{numerator} / 2^{k}`

        **2. Numerator in binary**

        `{sign}{bits}`

        **3. Shift {k} places left and normalize**

        `{sign}1.{fraction_bits} × 2^{exponent}`

        **Exact stored decimal**

        `{Decimal.from_float(float(stored))}`
        """)

    mo.hstack(
        [describe_stored(number.value, np.float32),
         describe_stored(number.value, np.float64)],
        widths="equal", gap=2, align="start",
    ).style({"overflow-wrap": "anywhere"})
    return


if __name__ == "__main__":
    app.run()

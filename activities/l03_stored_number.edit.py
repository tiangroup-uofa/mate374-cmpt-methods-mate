# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from decimal import Decimal
    from fractions import Fraction
    return Decimal, Fraction, mo, np


@app.cell
def _(mo):
    mo.md("""
    # What number did we actually store?

    Predict which inputs will be exact in both formats. Start with **0.75**,
    then try **0.1**, a large integer, and a very small number.
    The ratio describes the stored value, not necessarily the intended input.
    """)
    return


@app.cell
def _(mo):
    number = mo.ui.dropdown(
        options=["0.75", "0.1", "-13", "10240", "1e20", "1e-30"],
        value="0.75", label="Decimal input",
    )
    number
    return (number,)


@app.cell
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

        Short display: `{str(stored)}` · equals intended input exactly? **{exact}**

        **1. Stored ratio:** `{numerator} / {denominator}` = `{numerator} / 2^{k}`

        **2. Numerator in binary:** `{sign}{bits}`

        **3. Move the binary point {k} places left, then normalize:**

        `{sign}1.{fraction_bits} × 2^{exponent}`

        **Exact stored decimal:** `{Decimal.from_float(float(stored))}`
        """)

    mo.vstack([describe_stored(number.value, np.float32),
               describe_stored(number.value, np.float64)])
    return


@app.cell
def _(mo):
    mo.md("""
    **Check:** for 0.75 the ratio is 3/4, so the binary value is
    0.11 = 1.1 × 2⁻¹. For 0.1, count the significant bits in each format,
    starting with the leading 1. Why does printing more digits not repair
    the difference from the intended input?

    Open **Edit code** to inspect `as_integer_ratio()` and `bin()`.
    Large values can have denominator 1; small values can have a large
    power-of-two denominator. The normalization procedure is unchanged.
    """)
    return


if __name__ == "__main__":
    app.run()

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


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import struct
    from fractions import Fraction

    return Fraction, mo, np, struct


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # S02 · Number Converter and Floating-Point Dissector

    Use this tool to explore how numbers are converted between decimal and binary, and how IEEE 754 floating-point hardware stores them.

    1. **Integers**: Repeated division by 2.
    2. **Fractions**: Repeated multiplication by 2 (finite vs. repeating).
    3. **IEEE 754**: Sign bit, biased exponent, and stored fraction fields for `float32` and `float64`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    val_input = mo.ui.text(
        value="-13.625",
        label="Enter any decimal number (integer or float)",
    )
    precision_select = mo.ui.radio(
        options=["float32 (binary32, 24 bits sig)", "float64 (binary64, 53 bits sig)"],
        value="float32 (binary32, 24 bits sig)",
        label="IEEE Format",
        inline=True,
    )
    mo.hstack([val_input, precision_select], justify="start", align="center")
    return precision_select, val_input


@app.cell(hide_code=True)
def _(Fraction, np, precision_select, struct, val_input):
    raw_str = val_input.value.strip()
    is_float32 = "float32" in precision_select.value

    error_msg = None
    try:
        val = float(raw_str)
    except ValueError:
        val = None
        error_msg = f"Invalid decimal input: '{raw_str}'"

    steps_div = []
    steps_mul = []
    bits_str = ""
    sign_bit = "0"
    exp_bits = ""
    frac_bits = ""
    ratio_str = ""
    is_exact = False
    norm_str = ""
    biased_exp = 0
    actual_exp = 0
    bias = 127 if is_float32 else 1023
    fmt_char = "f" if is_float32 else "d"
    exp_len = 8 if is_float32 else 11
    frac_len = 23 if is_float32 else 52
    int_part = 0

    if val is not None:
        if is_float32:
            val_typed = np.float32(val)
            packed = struct.pack(">f", float(val_typed))
            ratio = float(val_typed).as_integer_ratio()
        else:
            val_typed = np.float64(val)
            packed = struct.pack(">d", float(val_typed))
            ratio = float(val_typed).as_integer_ratio()

        bits_str = "".join(f"{b:08b}" for b in packed)
        sign_bit = bits_str[0]
        exp_bits = bits_str[1 : 1 + exp_len]
        frac_bits = bits_str[1 + exp_len :]

        biased_exp = int(exp_bits, 2)
        actual_exp = biased_exp - bias

        ratio_str = f"{ratio[0]} / {ratio[1]}"
        try:
            is_exact = Fraction(raw_str) == Fraction(ratio[0], ratio[1])
        except Exception:
            is_exact = False

        abs_val = abs(val)
        int_part = int(abs_val)
        frac_part = abs_val - int_part

        curr = int_part
        if curr == 0:
            steps_div.append((0, 0, 0))
        else:
            while curr > 0:
                q = curr // 2
                r = curr % 2
                steps_div.append((curr, q, r))
                curr = q

        curr_f = frac_part
        idx = 0
        while curr_f > 0 and idx < 16:
            prod = curr_f * 2.0
            bit = int(prod)
            rem = prod - bit
            steps_mul.append((curr_f, prod, bit, rem))
            curr_f = rem
            idx += 1

        norm_sign = "-" if val < 0 else ""
        norm_str = f"{norm_sign}(1.{frac_bits})_2 * 2^({actual_exp})"

    return (
        actual_exp,
        bias,
        biased_exp,
        bits_str,
        error_msg,
        exp_bits,
        exp_len,
        fmt_char,
        frac_bits,
        frac_len,
        int_part,
        is_exact,
        is_float32,
        norm_str,
        precision_select,
        ratio_str,
        raw_str,
        sign_bit,
        steps_div,
        steps_mul,
        val,
        val_input,
    )


@app.cell(hide_code=True)
def _(
    actual_exp,
    bias,
    biased_exp,
    bits_str,
    error_msg,
    exp_bits,
    frac_bits,
    int_part,
    is_exact,
    is_float32,
    mo,
    norm_str,
    ratio_str,
    sign_bit,
    steps_div,
    steps_mul,
):
    if error_msg:
        out = mo.md(f"⚠️ **Error:** {error_msg}")
    else:
        div_rows = "\n".join(
            [f"| {curr} | {q} | **{r}** |" for (curr, q, r) in steps_div]
        )
        div_md = f"""
### 1. Integer Part: Repeated Division by 2
Absolute integer part: **{int_part}**

| Number | Quotient (/ 2) | Remainder |
|:---:|:---:|:---:|
{div_rows}

*Read remainders from bottom to top*: binary integer = `{''.join(str(r) for _, _, r in reversed(steps_div)) or '0'}`
"""

        if steps_mul:
            mul_rows = "\n".join(
                [f"| {f:.6g} | {p:.6g} | **{b}** | {r:.6g} |" for (f, p, b, r) in steps_mul]
            )
            mul_md = f"""
### 2. Fractional Part: Repeated Multiplication by 2
| Current | Product (* 2) | Integer bit | Remainder |
|:---:|:---:|:---:|:---:|
{mul_rows}

*Read bits from top to bottom*: binary fraction = `0.{''.join(str(b) for _, _, b, _ in steps_mul)}...`
"""
        else:
            mul_md = "### 2. Fractional Part\nNo fractional part (pure integer)."

        format_name = "binary32 (`float32`)" if is_float32 else "binary64 (`float64`)"
        exact_badge = "✅ **Exact representation**" if is_exact else "⚠️ **Rounded approximation** (repeating binary expansion)"

        ieee_md = f"""
### 3. IEEE 754 Hardware Encoding ({format_name})

- {exact_badge}
- **Stored integer ratio:** `{ratio_str}`
- **Normalized form:** `{norm_str}`
- **Sign ($s$):** `{sign_bit}` ({'-' if sign_bit == '1' else '+'})
- **Biased exponent ($E$):** `{biased_exp}` = actual exponent $p={actual_exp}$ + bias $b={bias}$
- **Exponent bits ({len(exp_bits)} bits):** `{exp_bits}`
- **Fraction / Mantissa bits ({len(frac_bits)} bits, leading 1 omitted):**
  `{frac_bits}`

#### Stored bit layout:
```text
Sign  | Exponent ({len(exp_bits)}b)  | Fraction ({len(frac_bits)}b)
  {sign_bit}   | {exp_bits} | {frac_bits[:12]}...{frac_bits[-6:]}
```
"""
        out = mo.vstack([mo.md(div_md), mo.md(mul_md), mo.md(ieee_md)])

    return (out,)


if __name__ == "__main__":
    app.run()

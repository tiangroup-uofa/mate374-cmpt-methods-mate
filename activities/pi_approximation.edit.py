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
    import struct
    import time

    import marimo as mo
    import numpy as np

    return mo, np, struct, time


@app.cell
def _(mo):
    mo.md(r"""
    # How quickly can we approximate π?

    A materials model may use π in a very ordinary formula, such as the
    cross-sectional area of a wire, grain, or cylindrical sample:

    $$A = \pi r^2.$$

    The exact value is available to us now, so π is a convenient test
    problem. We can ask three numerical questions:

    1. How does an algorithm turn a series into additions and loops?
    2. Does a different implementation change the time or the answer?
    3. How do we measure the error when a reference value is available?

    The first two implementations below calculate the same Leibniz series.
    The function signatures are supplied so you can edit the body rather
    than start with a blank programming exercise.
    """)
    return


@app.cell
def _(mo):
    terms = mo.ui.slider(
        start=10,
        stop=200_000,
        step=10,
        value=10_000,
        include_input=True,
        label="Number of terms, N",
        show_value=True,
    )
    method = mo.ui.dropdown(
        options={
            "Python loop · Leibniz": "loop",
            "NumPy array · Leibniz": "numpy",
            "Python loop · Nilakantha": "nilakantha",
        },
        value="Python loop · Leibniz",
        label="Implementation to benchmark",
    )
    mo.vstack([terms, method], align="start", gap=1)
    return method, terms


@app.cell
def _(np):
    def pi_leibniz_loop(n_terms):
        # Given algorithm: edit the indented body and keep the signature.
        total = 0.0
        for k in range(n_terms):
            sign = 1.0 if k % 2 == 0 else -1.0
            total += sign / (2.0 * k + 1.0)
        return 4.0 * total

    def pi_leibniz_numpy(n_terms):
        # Same series, expressed with an array operation.
        indices = np.arange(n_terms, dtype=np.float64)
        signs = np.where(indices % 2 == 0, 1.0, -1.0)
        return 4.0 * np.sum(signs / (2.0 * indices + 1.0))

    def pi_nilakantha_loop(n_terms):
        # A second series: it converges faster for this small experiment.
        total = 3.0
        for k in range(1, n_terms + 1):
            sign = 1.0 if k % 2 == 1 else -1.0
            denominator = (2.0 * k) * (2.0 * k + 1.0) * (2.0 * k + 2.0)
            total += 4.0 * sign / denominator
        return total

    return pi_leibniz_loop, pi_leibniz_numpy, pi_nilakantha_loop


@app.cell
def _(
    method,
    pi_leibniz_loop,
    pi_leibniz_numpy,
    pi_nilakantha_loop,
    terms,
    time,
):
    functions = {
        "loop": pi_leibniz_loop,
        "numpy": pi_leibniz_numpy,
        "nilakantha": pi_nilakantha_loop,
    }
    selected_function = functions[method.value]

    start = time.perf_counter()
    estimate = selected_function(int(terms.value))
    elapsed_seconds = time.perf_counter() - start

    absolute_error = abs(estimate - 3.141592653589793)
    relative_error = absolute_error / 3.141592653589793
    return absolute_error, elapsed_seconds, estimate, relative_error


@app.cell
def _(
    absolute_error,
    elapsed_seconds,
    estimate,
    method,
    mo,
    relative_error,
    terms,
):
    mo.callout(
        mo.md(
            fr"""
            ## One approximation

            **Selected implementation:** `{method.value}`  
            **Terms:** {int(terms.value):,}

            - estimate: **{estimate:.15f}**
            - absolute error: $|\pi_N-\pi| = $ **{absolute_error:.3e}**
            - relative error: $|\pi_N-\pi|/\pi = $ **{relative_error:.3e}**
            - one-call wall time: **{elapsed_seconds:.5f} s**

            Timing depends on the browser and machine. The error is a property
            of this approximation and the finite-precision arithmetic used to
            evaluate it.
            """
        ),
        kind="success" if relative_error < 1.0e-4 else "warn",
    )
    return


@app.cell
def _(mo, np, pi_leibniz_loop, pi_leibniz_numpy, pi_nilakantha_loop):
    def error_chart(sample_counts, curves):
        width, height = 700, 340
        left, right, top, bottom = 72, 18, 30, 58
        x_values = np.log10(sample_counts.astype(float))
        all_errors = np.concatenate([np.asarray(errors) for _, errors, _ in curves])
        y_values = np.log10(np.clip(all_errors, 1.0e-16, None))
        x_min, x_max = x_values.min(), x_values.max()
        y_min, y_max = y_values.min() - 0.25, y_values.max() + 0.25

        def x_pixel(value):
            return left + (value - x_min) / (x_max - x_min) * (width - left - right)

        def y_pixel(value):
            return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

        lines = []
        for label, errors, color in curves:
            points = " ".join(
                f"{x_pixel(x):.1f},{y_pixel(np.log10(max(error, 1.0e-16))):.1f}"
                for x, error in zip(x_values, errors)
            )
            lines.append(
                f'<polyline points="{points}" fill="none" '
                f'stroke="{color}" stroke-width="3" />'
            )

        x_ticks = "".join(
            f'<text x="{x_pixel(x):.1f}" y="{height - 35}" '
            f'text-anchor="middle" font-size="13">10^{int(x)}</text>'
            for x in x_values
        )
        y_tick_values = np.arange(np.ceil(y_min), np.floor(y_max) + 1)
        y_ticks = "".join(
            f'<text x="{left - 10}" y="{y_pixel(y) + 5:.1f}" '
            f'text-anchor="end" font-size="13">10^{int(y)}</text>'
            for y in y_tick_values
        )
        legend = " ".join(
            f'<tspan fill="{color}">● {label}</tspan>'
            for label, _, color in curves
        )
        return mo.Html(
            f"""
            <figure aria-label="Log-log chart of approximation error versus terms">
              <svg viewBox="0 0 {width} {height}" role="img"
                   style="max-width: 700px; width: 100%; height: auto;">
                <title>Approximation error decreases as the number of terms increases</title>
                <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"
                      stroke="currentColor" />
                <line x1="{left}" y1="{height - bottom}" x2="{width - right}"
                      y2="{height - bottom}" stroke="currentColor" />
                {x_ticks}
                {y_ticks}
                {''.join(lines)}
                <text x="{(left + width - right) / 2:.1f}" y="{height - 8}"
                      text-anchor="middle" font-size="14">number of terms, N</text>
                <text x="18" y="{(top + height - bottom) / 2:.1f}"
                      text-anchor="middle" font-size="14"
                      transform="rotate(-90 18 {(top + height - bottom) / 2:.1f})">absolute error</text>
                <text x="{left}" y="18" font-size="13">{legend}</text>
              </svg>
            </figure>
            """
        )

    sample_counts = np.array([10, 100, 1_000, 10_000, 100_000], dtype=int)
    loop_errors = np.array(
        [abs(pi_leibniz_loop(int(n)) - np.pi) for n in sample_counts]
    )
    numpy_errors = np.array(
        [abs(pi_leibniz_numpy(int(n)) - np.pi) for n in sample_counts]
    )
    nilakantha_errors = np.array(
        [abs(pi_nilakantha_loop(int(n)) - np.pi) for n in sample_counts]
    )
    error_chart(
        sample_counts,
        [
            ("Leibniz loop", loop_errors, "#007c41"),
            ("NumPy Leibniz", numpy_errors, "#d87700"),
            ("Nilakantha", nilakantha_errors, "#5b4b9a"),
        ],
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            ### What the chart is saying

            The Python loop and NumPy version use the same Leibniz series, so
            their errors nearly overlap. NumPy changes the implementation and
            usually the time, not the mathematical approximation. Nilakantha
            changes the series itself, so its error falls much faster here.

            More terms reduce **truncation error** at first. Eventually,
            **round-off error** from finite-precision arithmetic can limit or
            even spoil improvement. More computation is not automatically more
            truth.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    number = mo.ui.dropdown(
        options={
            "0.1": 0.1,
            "0.2": 0.2,
            "0.5": 0.5,
            "1.5": 1.5,
        },
        value="0.1",
        label="Inspect one stored float",
    )
    number
    return (number,)


@app.cell
def _(mo, number, struct):
    value = float(number.value)
    bits = f"{int.from_bytes(struct.pack('>d', value), 'big'):064b}"
    sign, exponent, fraction = bits[0], bits[1:12], bits[12:]
    numerator, denominator = value.as_integer_ratio()
    mo.md(
        f"""
        ### A quick floating-point warm-up

        The selected Python value is stored as an IEEE-754 double with
        **1 sign bit**, **11 exponent bits**, and **52 fraction bits**:

        ```text
        sign  exponent      fraction
          {sign}   {exponent}  {fraction}
        ```

        Python reports the exact stored value as
        `{numerator} / {denominator}`. For `0.1`, that fraction is not exactly
        one tenth. This is the beginning of the round-off story; the next
        lecture will make the representation and its consequences more precise.
        """
    )
    return


@app.cell
def _(mo):
    rounded_sum = 0.1 + 0.2
    roundoff_difference = rounded_sum - 0.3
    mo.callout(
        mo.md(
            f"""
            **A tiny round-off experiment:** `0.1 + 0.2` is
            `{rounded_sum:.17f}`, not exactly `0.3`. The difference
            `{roundoff_difference:.3e}` is small, but it is real in the stored
            floating-point arithmetic.
            """
        ),
        kind="warn" if roundoff_difference else "success",
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Try this in the editor

            Change the body of `pi_leibniz_loop` without changing its name or
            inputs. For example, replace the temporary `sign` variable with a
            direct expression, then run the cell and compare the estimate,
            error, and chart. A function is just a named recipe here; marimo
            tracks which later cells depend on it.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

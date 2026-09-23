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
    # Numerical integration: area under a stress–strain curve

    The energy density absorbed before failure is the area under a
    stress–strain curve:

    $$
    W=\int_0^{\varepsilon_f}\sigma(\varepsilon)\,\mathrm{d}\varepsilon.
    $$

    Real curves are tabulated measurements, so we approximate the area from
    discrete points. We will compare a left-rectangle rule, the trapezoidal
    rule, and Simpson's one-third rule. The synthetic model supplies an
    exact reference so that the numerical error is visible; real experiments
    would require a different verification argument.
    """)
    return


@app.cell
def _(mo):
    points = mo.ui.slider(
        start=5,
        stop=21,
        step=2,
        value=11,
        label="Number of equally spaced data points",
        show_value=True,
    )
    noise = mo.ui.dropdown(
        options={
            "clean curve": 0.0,
            "measurement noise": 0.01,
        },
        value="clean curve",
        label="Data quality",
    )
    mo.vstack([points, noise], align="start", gap=1)
    return noise, points


@app.cell
def _(np):
    def stress_model(strain):
        return 200.0 * strain + 1_000.0 * strain**2

    def left_rectangle(values, spacing):
        return spacing * np.sum(values[:-1])

    def trapezoidal(values, spacing):
        return spacing * (
            0.5 * values[0] + np.sum(values[1:-1]) + 0.5 * values[-1]
        )

    def simpson_one_third(values, spacing):
        # Given function: Simpson requires an even number of intervals.
        odd_terms = np.sum(values[1:-1:2])
        even_terms = np.sum(values[2:-1:2])
        return spacing / 3.0 * (
            values[0] + values[-1] + 4.0 * odd_terms + 2.0 * even_terms
        )

    return left_rectangle, simpson_one_third, stress_model, trapezoidal


@app.cell
def _(
    left_rectangle,
    noise,
    np,
    points,
    simpson_one_third,
    stress_model,
    trapezoidal,
):
    strains = np.linspace(0.0, 0.1, int(points.value))
    values = stress_model(strains) + noise.value * np.sin(31.0 * strains)
    spacing = strains[1] - strains[0]
    exact_integral = 100.0 * 0.1**2 + (1_000.0 / 3.0) * 0.1**3
    estimates = np.array(
        [
            left_rectangle(values, spacing),
            trapezoidal(values, spacing),
            simpson_one_third(values, spacing),
        ]
    )
    errors = estimates - exact_integral
    return errors, estimates, exact_integral


@app.cell
def _(errors, estimates, exact_integral, mo, noise, points):
    labels = ["left rectangle", "trapezoid", "Simpson 1/3"]
    rows = "\n".join(
        f"| {label} | {estimate:.6f} | {error:.3e} |"
        for label, estimate, error in zip(labels, estimates, errors)
    )
    mo.callout(
        mo.md(
            f"""
            ## Toughness estimate

            Exact synthetic reference: **{exact_integral:.6f} GPa**  
            This has the same numerical units as **GJ/m³** because strain is
            dimensionless.  
            Points: **{points.value}** · Data: **{noise.value}**

            | Rule | estimated area (GPa) | error (GPa) |
            |---|---:|---:|
            {rows}

            Simpson's rule is exact for this quadratic stress model when the
            spacing and point-count conditions are satisfied. Noise changes the
            comparison: a higher-order rule cannot recover information absent
            from the measurements.
            """
        ),
        kind="success" if noise.value == 0.0 else "info",
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Try this in the editor

            Change `simpson_one_third` while keeping the input array and spacing.
            Then change the number of points and turn on noise. Before using a
            formula, check its assumptions: equal spacing and an even number of
            intervals for composite Simpson 1/3.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

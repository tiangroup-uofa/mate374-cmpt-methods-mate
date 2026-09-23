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
    # Numerical differentiation: rate of change versus noise

    A stress–strain experiment records an energy density $u(\varepsilon)$.
    The stress is the derivative

    $$
    \sigma(\varepsilon)=\frac{\mathrm{d}u}{\mathrm{d}\varepsilon}.
    $$

    When $u$ is available only at discrete points, we estimate the derivative
    with neighboring values. Forward and backward differences are first
    order; the centered difference is second order for equally spaced,
    smooth data. Experimental noise changes the trade-off: smaller spacing
    can expose more noise rather than more truth.
    """)
    return


@app.cell
def _(mo):
    spacing = mo.ui.dropdown(
        options={
            "h = 0.005": 0.005,
            "h = 0.002": 0.002,
            "h = 0.001": 0.001,
        },
        value="h = 0.002",
        label="Strain spacing h",
    )
    target_strain = mo.ui.slider(
        start=0.004,
        stop=0.016,
        step=0.002,
        value=0.010,
        label="Strain at which to estimate stress",
        show_value=True,
    )
    noise = mo.ui.dropdown(
        options={
            "clean model values": 0.0,
            "measurement noise": 0.002,
        },
        value="clean model values",
        label="Data quality",
    )
    mo.vstack([spacing, target_strain, noise], align="start", gap=1)
    return noise, spacing, target_strain


@app.cell
def _():
    ELASTIC_MODULUS = 200.0  # GPa
    CUBIC_COEFFICIENT = 1_000.0  # GPa

    def energy_density(strain):
        return (
            0.5 * ELASTIC_MODULUS * strain**2
            + CUBIC_COEFFICIENT * strain**3
        )

    def exact_stress(strain):
        return ELASTIC_MODULUS * strain + 3.0 * CUBIC_COEFFICIENT * strain**2

    def finite_difference_estimates(values, index, step):
        # Given function: keep the three local slope formulas visible.
        forward = (values[index + 1] - values[index]) / step
        backward = (values[index] - values[index - 1]) / step
        central = (values[index + 1] - values[index - 1]) / (2.0 * step)
        return forward, backward, central

    return energy_density, exact_stress, finite_difference_estimates


@app.cell
def _(
    energy_density,
    exact_stress,
    finite_difference_estimates,
    noise,
    np,
    spacing,
    target_strain,
):
    step = float(spacing.value)
    strains = np.arange(0.0, 0.0200001, step)
    clean_values = energy_density(strains)
    noisy_values = clean_values + noise.value * np.sin(17.0 * strains / 0.02)
    target_index = int(round(target_strain.value / step))
    target = strains[target_index]
    forward, backward, central = finite_difference_estimates(
        noisy_values, target_index, step
    )
    exact = exact_stress(target)
    estimates = np.array([forward, backward, central])
    errors = estimates - exact
    return errors, estimates, exact, target


@app.cell
def _(errors, estimates, exact, mo, noise, spacing, target):
    labels = ["forward", "backward", "central"]
    rows = "\n".join(
        f"| {label} | {estimate:.6f} | {error:.3e} |"
        for label, estimate, error in zip(labels, estimates, errors)
    )
    mo.callout(
        mo.md(
            f"""
            ## Stress estimate at ε = {target:.3f}

            Exact model stress: **{exact:.6f} GPa**  
            Spacing: **{spacing.value}**  
            Data: **{noise.value}**

            | Formula | estimated stress (GPa) | error (GPa) |
            |---|---:|---:|
            {rows}

            The centered formula usually wins for clean, equally spaced data.
            With noise, reducing the spacing can amplify point-to-point
            fluctuations. The derivative method and data quality must be reported
            together.
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
            ### Try this in the editor

            Change `finite_difference_estimates` while preserving its inputs and
            outputs. Compare the centered estimate with the exact model, then
            turn on measurement noise. Which apparent improvement is actually a
            discretization improvement, and which is only sensitivity to the
            data?
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

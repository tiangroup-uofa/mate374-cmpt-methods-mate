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
    # Root-finding methods: robust or fast?

    We keep the same thermal-balance model and change only the numerical
    method. There are two broad strategies:

    - **bracketing methods** preserve an interval containing a root;
    - **open methods** use local information and can converge quickly, but
      may leave the physical domain or diverge.

    We will compare bisection, Newton, secant, and fixed-point iteration.
    The point is not to memorize four code templates. It is to match a
    method to the information and reliability the materials problem allows.
    """)
    return


@app.cell
def _(mo):
    heat_flux = mo.ui.slider(
        start=2_000,
        stop=20_000,
        step=1_000,
        value=10_000,
        label="Incoming heat flux, q″ (W/m²)",
        show_value=True,
    )
    initial_guess = mo.ui.slider(
        start=350,
        stop=1_100,
        step=25,
        value=500,
        label="Newton / fixed-point start, T₀ (K)",
        show_value=True,
    )
    second_guess = mo.ui.slider(
        start=400,
        stop=1_100,
        step=25,
        value=700,
        label="Secant second start, T₁ (K)",
        show_value=True,
    )
    gain = mo.ui.slider(
        start=0.005,
        stop=0.025,
        step=0.005,
        value=0.015,
        label="Fixed-point gain (K·m²/W)",
        show_value=True,
    )
    mo.vstack([heat_flux, initial_guess, second_guess, gain], align="start", gap=1)
    return gain, heat_flux, initial_guess, second_guess


@app.cell
def _():
    STEFAN_BOLTZMANN = 5.670374419e-8
    CONVECTION_COEFFICIENT = 15.0
    AMBIENT_TEMPERATURE = 300.0
    EMISSIVITY = 0.80

    def heat_balance_residual(surface_temperature_K, heat_flux):
        convection = CONVECTION_COEFFICIENT * (
            surface_temperature_K - AMBIENT_TEMPERATURE
        )
        radiation = EMISSIVITY * STEFAN_BOLTZMANN * (
            surface_temperature_K**4 - AMBIENT_TEMPERATURE**4
        )
        return convection + radiation - heat_flux

    def heat_balance_derivative(surface_temperature_K):
        return CONVECTION_COEFFICIENT + 4.0 * EMISSIVITY * STEFAN_BOLTZMANN * (
            surface_temperature_K**3
        )

    def bisection_root(residual, left, right, tolerance, max_iterations):
        left_value = residual(left)
        right_value = residual(right)
        history = []
        if left_value * right_value > 0.0:
            return float("nan"), history, "invalid bracket"
        for iteration in range(1, max_iterations + 1):
            midpoint = 0.5 * (left + right)
            midpoint_value = residual(midpoint)
            history.append((iteration, midpoint, midpoint_value))
            if abs(midpoint_value) <= tolerance or right - left <= tolerance:
                return midpoint, history, "converged"
            if left_value * midpoint_value <= 0.0:
                right = midpoint
                right_value = midpoint_value
            else:
                left = midpoint
                left_value = midpoint_value
        return midpoint, history, "iteration limit"

    def newton_root(residual, derivative, initial, tolerance, max_iterations):
        current = initial
        history = []
        for iteration in range(1, max_iterations + 1):
            if current <= 0.0:
                return float("nan"), history, "left physical domain"
            value = residual(current)
            history.append((iteration, current, value))
            slope = derivative(current)
            if abs(slope) < 1.0e-12:
                return current, history, "zero derivative"
            if abs(value) <= tolerance:
                return current, history, "converged"
            current = current - value / slope
        return current, history, "iteration limit"

    def secant_root(residual, first, second, tolerance, max_iterations):
        history = []
        for iteration in range(1, max_iterations + 1):
            if first <= 0.0 or second <= 0.0:
                return float("nan"), history, "left physical domain"
            first_value = residual(first)
            second_value = residual(second)
            history.append((iteration, second, second_value))
            denominator = second_value - first_value
            if abs(denominator) < 1.0e-12:
                return second, history, "zero secant slope"
            if abs(second_value) <= tolerance:
                return second, history, "converged"
            next_value = second - second_value * (second - first) / denominator
            first, second = second, next_value
        return second, history, "iteration limit"

    def fixed_point_root(residual, initial, gain, tolerance, max_iterations):
        current = initial
        history = []
        for iteration in range(1, max_iterations + 1):
            if current <= 0.0:
                return float("nan"), history, "left physical domain"
            value = residual(current)
            history.append((iteration, current, value))
            if abs(value) <= tolerance:
                return current, history, "converged"
            current = current - gain * value
        return current, history, "iteration limit"

    return (
        AMBIENT_TEMPERATURE,
        bisection_root,
        fixed_point_root,
        heat_balance_derivative,
        heat_balance_residual,
        newton_root,
        secant_root,
    )


@app.cell
def _(
    AMBIENT_TEMPERATURE,
    bisection_root,
    fixed_point_root,
    gain,
    heat_balance_derivative,
    heat_balance_residual,
    heat_flux,
    initial_guess,
    newton_root,
    np,
    secant_root,
    second_guess,
):
    tolerance = 1.0e-3

    def residual(temperature):
        return heat_balance_residual(temperature, heat_flux.value)

    methods = [
        (
            "Bisection",
            bisection_root(
                residual, AMBIENT_TEMPERATURE, 1_200.0, tolerance, 60
            ),
        ),
        (
            "Newton",
            newton_root(
                residual,
                heat_balance_derivative,
                initial_guess.value,
                tolerance,
                30,
            ),
        ),
        (
            "Secant",
            secant_root(
                residual,
                initial_guess.value,
                second_guess.value,
                tolerance,
                30,
            ),
        ),
        (
            "Fixed point",
            fixed_point_root(
                residual, initial_guess.value, gain.value, tolerance, 60
            ),
        ),
    ]

    rows = []
    for _method_name, (_root, _history, _status) in methods:
        residual_value = residual(_root) if np.isfinite(_root) else float("nan")
        root_text = f"{_root:.4f} K" if np.isfinite(_root) else "—"
        residual_text = (
            f"{residual_value:.2e} W/m²" if np.isfinite(_root) else "—"
        )
        rows.append(
            f"| {_method_name} | {_status} | {len(_history)} | {root_text} | {residual_text} |"
        )
    table = "\n".join(rows)
    return methods, table


@app.cell
def _(mo, table):
    mo.md(
        f"""
        ## Method comparison

        | Method | Status | Iterations | Root | Residual |
        |---|---|---:|---:|---:|
        {table}

        Bisection spends iterations protecting the bracket. Newton uses a
        derivative and is usually fast near the root. Secant avoids an explicit
        derivative but still relies on good starting values. Fixed-point
        iteration depends strongly on the chosen iteration map and gain.
        """
    )
    return


@app.cell
def _(methods, mo, np):
    colors = {
        "Bisection": "#007c41",
        "Newton": "#d87700",
        "Secant": "#5b4b9a",
        "Fixed point": "#007ea7",
    }
    all_values = [
        abs(row[2])
        for _, (_, history, _) in methods
        for row in history
        if np.isfinite(row[2]) and row[2] != 0.0
    ]
    if all_values:
        y_min = max(min(all_values), 1.0e-12)
        y_max = max(max(all_values), y_min * 10.0)
        y_range = max(np.log10(y_max) - np.log10(y_min), 1.0)
        svg_lines = []
        for _method_name, (_, _history, _) in methods:
            if not _history:
                continue
            values = [max(abs(row[2]), 1.0e-12) for row in _history]
            points = " ".join(
                f"{40 + i / max(len(values) - 1, 1) * 630:.1f},"
                f"{25 + (np.log10(y_max) - np.log10(value)) / y_range * 240:.1f}"
                for i, value in enumerate(values)
            )
            svg_lines.append(
                f'<polyline points="{points}" fill="none" '
                f'stroke="{colors[_method_name]}" stroke-width="3" />'
            )
        chart = mo.Html(
            f"""
            <figure aria-label="Residual magnitude by iteration for root methods">
              <svg viewBox="0 0 700 300" role="img"
                   style="max-width: 700px; width: 100%; height: auto;">
                <title>Root-method residual histories</title>
                {''.join(svg_lines)}
                <line x1="40" y1="265" x2="670" y2="265" stroke="currentColor" />
                <line x1="40" y1="25" x2="40" y2="265" stroke="currentColor" />
                <text x="355" y="292" text-anchor="middle" font-size="14">iteration</text>
                <text x="14" y="145" text-anchor="middle" font-size="14"
                      transform="rotate(-90 14 145)">|residual| (log scale)</text>
              </svg>
            </figure>
            """
        )
    else:
        chart = mo.md("No residual history is available for the selected starts.")
    chart
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Interpretation prompt

            Which method would you use if the calculation were run once and
            reliability mattered most? Which would you use inside a large
            parameter sweep? Defend the choice with the bracket, derivative,
            starting guesses, iteration count, and residual—not speed alone.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

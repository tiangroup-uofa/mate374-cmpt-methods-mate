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
    # Roots versus minima in a regular solution

    A stationary point of a free energy satisfies $\mathrm{d}g/\mathrm{d}x=0$,
    but not every stationary point is a stable equilibrium. For a symmetric
    binary regular solution,

    $$
    g(x)=RT[x\ln x+(1-x)\ln(1-x)]+\Omega x(1-x).
    $$

    We compare the stationary roots of $g'(x)=0$ with the curvature and free
    energy at those roots. A root finder answers one question; stability and
    global preference require additional evidence.
    """)
    return


@app.cell
def _(mo):
    temperature = mo.ui.slider(
        start=250,
        stop=800,
        step=10,
        value=400,
        label="Temperature T (K)",
        show_value=True,
    )
    mo.vstack([temperature], align="start")
    return (temperature,)


@app.cell
def _(np):
    GAS_CONSTANT = 8.314
    INTERACTION_ENERGY = 8_000.0

    def free_energy(composition, temperature):
        x = np.asarray(composition)
        safe_x = np.clip(x, 1.0e-8, 1.0 - 1.0e-8)
        mixing = safe_x * np.log(safe_x) + (1.0 - safe_x) * np.log(1.0 - safe_x)
        return GAS_CONSTANT * temperature * mixing + INTERACTION_ENERGY * safe_x * (1.0 - safe_x)

    def free_energy_derivative(composition, temperature):
        x = np.clip(composition, 1.0e-8, 1.0 - 1.0e-8)
        return GAS_CONSTANT * temperature * np.log(x / (1.0 - x)) + INTERACTION_ENERGY * (1.0 - 2.0 * x)

    def free_energy_curvature(composition, temperature):
        x = np.clip(composition, 1.0e-8, 1.0 - 1.0e-8)
        return GAS_CONSTANT * temperature * (1.0 / x + 1.0 / (1.0 - x)) - 2.0 * INTERACTION_ENERGY

    return free_energy, free_energy_curvature, free_energy_derivative


@app.cell
def _(
    free_energy,
    free_energy_curvature,
    free_energy_derivative,
    np,
    temperature,
):
    compositions = np.linspace(1.0e-3, 1.0 - 1.0e-3, 1200)
    derivative_values = free_energy_derivative(compositions, temperature.value)
    roots = []

    def bisect(left, right):
        left_value = free_energy_derivative(left, temperature.value)
        for _ in range(60):
            middle = 0.5 * (left + right)
            middle_value = free_energy_derivative(middle, temperature.value)
            if abs(middle_value) < 1.0e-8:
                return middle
            if left_value * middle_value <= 0.0:
                right = middle
            else:
                left = middle
                left_value = middle_value
        return 0.5 * (left + right)

    for index in range(len(compositions) - 1):
        left, right = compositions[index], compositions[index + 1]
        left_value, right_value = derivative_values[index], derivative_values[index + 1]
        if abs(left_value) < 1.0e-6:
            roots.append(left)
        if left_value * right_value < 0.0:
            roots.append(bisect(left, right))
    if abs(free_energy_derivative(0.5, temperature.value)) < 1.0e-8:
        roots.append(0.5)
    roots = np.array(sorted(set(round(float(root), 10) for root in roots)))
    energies = free_energy(roots, temperature.value)
    curvatures = free_energy_curvature(roots, temperature.value)
    return compositions, curvatures, energies, roots


@app.cell
def _(compositions, free_energy, mo, roots, temperature):
    values = free_energy(compositions, temperature.value)
    x_pixels = 55 + (compositions - compositions.min()) / (compositions.max() - compositions.min()) * 610
    y_min, y_max = float(values.min()), float(values.max())
    y_pixels = 25 + (y_max - values) / max(y_max - y_min, 1.0) * 230
    curve_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(x_pixels, y_pixels))
    root_points = " ".join(
        f"{55 + (root - compositions.min()) / (compositions.max() - compositions.min()) * 610:.1f},"
        f"{25 + (y_max - float(free_energy(root, temperature.value))) / max(y_max - y_min, 1.0) * 230:.1f}"
        for root in roots
    )
    mo.Html(
        f"""
        <figure aria-label="Regular-solution free energy curve and stationary points">
          <svg viewBox="0 0 700 290" role="img" style="max-width:700px;width:100%;height:auto;">
            <title>Free energy as a function of composition</title>
            <polyline points="{curve_points}" fill="none" stroke="#007c41" stroke-width="3" />
            <polyline points="{root_points}" fill="none" stroke="#d87700" stroke-width="0" />
            <line x1="55" y1="255" x2="665" y2="255" stroke="currentColor" />
            <line x1="55" y1="25" x2="55" y2="255" stroke="currentColor" />
            <text x="360" y="282" text-anchor="middle" font-size="14">composition x</text>
            <text x="16" y="140" text-anchor="middle" font-size="14" transform="rotate(-90 16 140)">g(x) (J/mol)</text>
          </svg>
        </figure>
        """
    )
    return


@app.cell
def _(curvatures, energies, mo, roots, temperature):
    rows = "\n".join(
        f"| {root:.6f} | {energy:.3f} | {curvature:.3f} | {'local minimum' if curvature > 0 else 'local maximum'} |"
        for root, energy, curvature in zip(roots, energies, curvatures)
    )
    mo.callout(
        mo.md(
            f"""
            ## Stationary points at T = {temperature.value} K

            | composition x | g(x) (J/mol) | g″(x) | interpretation |
            |---:|---:|---:|---|
            {rows if len(roots) else '| — | — | — | no root found on the sampled domain |'}

            A root of $g'(x)$ is only a candidate state. Positive curvature
            indicates a local minimum; negative curvature indicates a local
            maximum. Comparing free-energy values is needed to discuss global
            preference.
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

            Change the body of `free_energy` while keeping the model inputs and
            units. Vary temperature through the critical regime. Which roots
            disappear, and why is solving `g'(x)=0` alone not enough to identify
            a stable phase?
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

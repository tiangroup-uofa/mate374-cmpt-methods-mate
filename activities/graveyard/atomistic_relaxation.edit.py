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
    # Atomistic relaxation: descend an energy landscape

    A two-atom toy material has a Lennard–Jones pair energy,

    $$
    E(r)=4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}
    -\left(\frac{\sigma}{r}\right)^6\right].
    $$

    The equilibrium separation is the minimum of $E(r)$, not merely any
    zero of its derivative. We use gradient descent as a transparent
    relaxation method and monitor energy, separation, and the force-like
    gradient.
    """)
    return


@app.cell
def _(mo):
    initial_separation = mo.ui.slider(
        start=0.80,
        stop=2.00,
        step=0.05,
        value=1.50,
        label="Initial separation r/σ",
        show_value=True,
    )
    learning_rate = mo.ui.slider(
        start=0.001,
        stop=0.020,
        step=0.001,
        value=0.010,
        label="Relaxation step size",
        show_value=True,
    )
    steps = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=60,
        label="Relaxation steps",
        show_value=True,
    )
    mo.vstack([initial_separation, learning_rate, steps], align="start", gap=1)
    return initial_separation, learning_rate, steps


@app.cell
def _(np):
    def lj_energy(separation):
        ratio = 1.0 / separation
        return 4.0 * (ratio**12 - ratio**6)

    def lj_gradient(separation):
        ratio = 1.0 / separation
        return 24.0 * (ratio**7 - 2.0 * ratio**13)

    def relax_separation(initial, learning_rate, n_steps):
        # Given function: inspect the energy-decreasing update.
        separations = np.zeros(n_steps + 1)
        energies = np.zeros(n_steps + 1)
        gradients = np.zeros(n_steps + 1)
        separations[0] = initial
        energies[0] = lj_energy(initial)
        gradients[0] = lj_gradient(initial)
        for step in range(n_steps):
            candidate = separations[step] - learning_rate * gradients[step]
            separations[step + 1] = max(candidate, 0.60)
            energies[step + 1] = lj_energy(separations[step + 1])
            gradients[step + 1] = lj_gradient(separations[step + 1])
        return separations, energies, gradients

    return (relax_separation,)


@app.cell
def _(initial_separation, learning_rate, relax_separation, steps):
    separations, energies, gradients = relax_separation(
        initial_separation.value, learning_rate.value, int(steps.value)
    )
    equilibrium_separation = 2.0 ** (1.0 / 6.0)
    return energies, equilibrium_separation, gradients, separations


@app.cell
def _(energies, equilibrium_separation, gradients, mo, separations):
    final_separation = separations[-1]
    final_energy = energies[-1]
    mo.callout(
        mo.md(
            f"""
            ## Relaxation result

            - analytic minimum: **r/σ = {equilibrium_separation:.6f}**
            - final separation: **r/σ = {final_separation:.6f}**
            - final energy: **E/ε = {final_energy:.6f}**
            - final gradient: **dE/dr = {gradients[-1]:.3e}**
            - energy decreased on every recorded step: **{bool((energies[1:] <= energies[:-1] + 1.0e-12).all())}**

            A converged optimizer should satisfy both a small gradient and a
            physically meaningful state. A small step change alone is not enough.
            """
        ),
        kind="success" if abs(gradients[-1]) < 0.1 else "warn",
    )
    return


@app.cell
def _(energies, mo, np, separations):
    x = np.arange(len(separations), dtype=float)
    x_pixels = 55 + x / max(x[-1], 1.0) * 610
    y_min, y_max = float(energies.min()), float(energies.max())
    y_pixels = 25 + (y_max - energies) / max(y_max - y_min, 1.0) * 230
    points = " ".join(f"{xv:.1f},{yv:.1f}" for xv, yv in zip(x_pixels, y_pixels))
    mo.Html(
        f"""
        <figure aria-label="Lennard-Jones energy during relaxation">
          <svg viewBox="0 0 700 290" role="img" style="max-width:700px;width:100%;height:auto;">
            <title>Energy decreases during atomistic relaxation</title>
            <polyline points="{points}" fill="none" stroke="#007c41" stroke-width="3" />
            <line x1="55" y1="255" x2="665" y2="255" stroke="currentColor" />
            <line x1="55" y1="25" x2="55" y2="255" stroke="currentColor" />
            <text x="360" y="282" text-anchor="middle" font-size="14">relaxation step</text>
            <text x="16" y="140" text-anchor="middle" font-size="14" transform="rotate(-90 16 140)">energy E/ε</text>
          </svg>
        </figure>
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Try this in the editor

            Change `relax_separation` or increase the learning rate. Watch for
            overshoot, a stuck lower bound, or a final state with a non-small
            gradient. This is the same verification habit used for root finding:
            inspect the path, not only the last number.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

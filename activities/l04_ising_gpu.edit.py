# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
#     "matplotlib>=3.9",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Microstructure evolution: 2D Ising model

        In materials science, the **2D Ising model** describes order–disorder transformations, magnetic domains, and phase separation. Each site on an $L \times L$ lattice holds a spin $s_i \in \{+1, -1\}$ (representing two magnetic orientations or binary atomic species A and B).

        The energy of the configuration is:
        $$
        E = -J \sum_{\langle i, j \rangle} s_i s_j
        $$
        At high temperature $T$, thermal fluctuations disorder the system. Below the critical Curie temperature:
        $$
        T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2.269 \frac{J}{k_B}
        $$
        spontaneous magnetization and domain coarsening occur!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    temperature_slider = mo.ui.slider(
        start=0.5,
        stop=5.0,
        step=0.1,
        value=2.27,
        label="Temperature T (J/kB, Tc ≈ 2.27)",
    )
    steps_slider = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=50,
        label="Monte Carlo steps",
    )
    lattice_size = mo.ui.dropdown(
        options={"100 × 100": 100, "200 × 200": 200},
        value="100 × 100",
        label="Lattice grid size L",
    )
    reset_button = mo.ui.button(label="Randomize Spins", value=0)

    mo.hstack([temperature_slider, steps_slider, lattice_size, reset_button], justify="start", align="center")
    return lattice_size, reset_button, steps_slider, temperature_slider


@app.cell(hide_code=True)
def _(lattice_size, np, reset_button, steps_slider, temperature_slider):
    # Reference dependencies so reactive recomputation happens
    _ = reset_button.value
    L = int(lattice_size.value)
    T = float(temperature_slider.value)
    num_steps = int(steps_slider.value)
    beta = 1.0 / max(T, 1e-4)

    # Initialize random lattice of +/- 1 spins
    rng = np.random.default_rng(42 + int(reset_button.value))
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

    # Checkerboard vectorized Metropolis updates
    # Even sites (i+j)%2 == 0 and odd sites (i+j)%2 == 1 are conditionally independent!
    # This checkerboard decomposition allows embarrassingly parallel GPU/vectorized array updates.
    row_idx, col_idx = np.indices((L, L))
    mask_even = (row_idx + col_idx) % 2 == 0
    mask_odd = ~mask_even

    for _ in range(num_steps):
        for current_mask in [mask_even, mask_odd]:
            # Sum 4 nearest neighbors with periodic boundary conditions
            neighbors = (
                np.roll(spins, 1, axis=0)
                + np.roll(spins, -1, axis=0)
                + np.roll(spins, 1, axis=1)
                + np.roll(spins, -1, axis=1)
            )
            # Change in energy if spin is flipped: dE = 2 * s_i * sum_neighbors
            dE = 2 * spins * neighbors
            # Metropolis criterion: accept if dE <= 0 or with probability exp(-beta * dE)
            prob = np.exp(-beta * dE)
            accept = (dE <= 0) | (rng.random(size=(L, L)) < prob)
            flip_mask = current_mask & accept
            spins[flip_mask] = -spins[flip_mask]

    magnetization = float(np.mean(spins))
    return L, T, magnetization, num_steps, spins


@app.cell(hide_code=True)
def _(L, T, magnetization, mo, num_steps, plt, spins):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(spins, cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
    ax.set_title(
        f"2D Ising domain structure: L={L}, T={T:.2f} J/kB\n"
        f"{num_steps} checkerboard MC sweeps, <M> = {magnetization:+.3f}",
        fontsize=11,
    )
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, 1])
    cbar.set_ticklabels(["Spin down (-1)", "Spin up (+1)"])
    fig.tight_layout()

    status = mo.md(
        f"""
        **State summary**: 
        - Temperature: **{T:.2f}** (Curie $T_c \\approx 2.27$)
        - Net magnetization $\\langle M \\rangle$: **{magnetization:+.3f}**
        - Phase: **{'Disordered paramagnetic (random fluctuations)' if T > 2.35 else 'Ordered ferromagnetic (macroscopic domains coarsening)'}**
        
        *Notice how checkerboard sublattices decouple updates: this is why GPUs and vectorized code accelerate grid simulations by $100\\times$ or more.*
        """
    )

    mo.vstack([fig, status])
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
#     "matplotlib>=3.9",
#     "torch>=2.2.0",
# ]
# ///
"""Molab GPU Accelerated 2D Ising / Potts Model.
Open directly in Molab with GPU runtime enabled.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    try:
        import torch
        has_torch = True
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    except ImportError:
        has_torch = False
        device = "cpu"

    return device, has_torch, mo, np, plt, time, torch


@app.cell(hide_code=True)
def _(device, has_torch, mo):
    mo.md(
        f"""
        # Massive Grid Simulation: 2D Ising Model on GPU

        - **PyTorch accelerator detected:** `{has_torch}`
        - **Active compute device:** **`{device}`**

        In materials computational engineering, domain growth, grain boundary evolution, and order–disorder transformations involve millions of lattice sites. GPUs execute thousands of local updates simultaneously using checkerboard decomposition.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    grid_slider = mo.ui.dropdown(
        options={"512 × 512 (262k sites)": 512, "1024 × 1024 (1M sites)": 1024, "2048 × 2048 (4.2M sites)": 2048},
        value="1024 × 1024 (1M sites)",
        label="Lattice size L",
    )
    temp_slider = mo.ui.slider(
        start=0.5,
        stop=4.5,
        step=0.1,
        value=2.27,
        label="Temperature T (Curie Tc ≈ 2.27)",
    )
    sweeps_slider = mo.ui.slider(
        start=10,
        stop=500,
        step=20,
        value=100,
        label="Monte Carlo sweeps",
    )
    run_btn = mo.ui.button(label="Simulate Microstructure", value=0)

    mo.hstack([grid_slider, temp_slider, sweeps_slider, run_btn], justify="start", align="center")
    return grid_slider, run_btn, sweeps_slider, temp_slider


@app.cell(hide_code=True)
def _(
    device,
    grid_slider,
    has_torch,
    np,
    run_btn,
    sweeps_slider,
    temp_slider,
    time,
    torch,
):
    _ = run_btn.value
    L = int(grid_slider.value)
    T = float(temp_slider.value)
    sweeps = int(sweeps_slider.value)
    beta = 1.0 / max(T, 1e-4)

    t0 = time.perf_counter()

    if has_torch and device in ["cuda", "mps"]:
        dev = torch.device(device)
        # Random initial spins in {-1, 1}
        spins = torch.where(torch.rand((L, L), device=dev) > 0.5, 1.0, -1.0)

        # Precompute checkerboard masks
        row = torch.arange(L, device=dev).unsqueeze(1)
        col = torch.arange(L, device=dev).unsqueeze(0)
        mask_even = ((row + col) % 2) == 0
        mask_odd = ~mask_even

        for _ in range(sweeps):
            for mask in [mask_even, mask_odd]:
                neighbors = (
                    torch.roll(spins, 1, dims=0)
                    + torch.roll(spins, -1, dims=0)
                    + torch.roll(spins, 1, dims=1)
                    + torch.roll(spins, -1, dims=1)
                )
                dE = 2.0 * spins * neighbors
                prob = torch.exp(-beta * dE)
                accept = (dE <= 0) | (torch.rand((L, L), device=dev) < prob)
                flip = mask & accept
                spins[flip] = -spins[flip]

        if dev.type == "cuda":
            torch.cuda.synchronize()
        runtime = time.perf_counter() - t0
        spins_cpu = spins.detach().cpu().numpy()
    else:
        # NumPy fallback
        rng = np.random.default_rng(42)
        spins_cpu = rng.choice(np.array([-1.0, 1.0]), size=(L, L))
        row, col = np.indices((L, L))
        mask_even = (row + col) % 2 == 0
        mask_odd = ~mask_even

        for _ in range(min(sweeps, 50)): # limit cpu sweeps
            for mask in [mask_even, mask_odd]:
                neighbors = (
                    np.roll(spins_cpu, 1, axis=0)
                    + np.roll(spins_cpu, -1, axis=0)
                    + np.roll(spins_cpu, 1, axis=1)
                    + np.roll(spins_cpu, -1, axis=1)
                )
                dE = 2.0 * spins_cpu * neighbors
                prob = np.exp(-beta * dE)
                accept = (dE <= 0) | (rng.random(size=(L, L)) < prob)
                flip = mask & accept
                spins_cpu[flip] = -spins_cpu[flip]

        runtime = time.perf_counter() - t0

    net_mag = float(np.mean(spins_cpu))
    total_updates = L * L * sweeps
    mcups = (total_updates / runtime) / 1e6 # Million lattice updates per second

    return L, T, mcups, net_mag, runtime, spins_cpu, sweeps


@app.cell(hide_code=True)
def _(L, T, mcups, mo, net_mag, plt, runtime, spins_cpu, sweeps):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(spins_cpu, cmap="copper", origin="lower")
    ax.set_title(
        f"2D Ising Domain Coarsening ({L}×{L})\n"
        f"T = {T:.2f} J/kB | {sweeps} sweeps | Net Mag: {net_mag:+.3f}",
        fontsize=11,
    )
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    report = mo.md(
        f"""
        ### Computational Performance:
        - **Execution time:** `{runtime:.3f}` seconds
        - **Lattice update throughput:** `{mcups:.1f}` Million spin updates/second (MCUPS)
        - **Physical state:** {'Disordered paramagnetic phase (above Tc)' if T > 2.35 else 'Coarsened ferromagnetic domains (below Tc)'}
        """
    )

    mo.vstack([fig, report])
    return


if __name__ == "__main__":
    app.run()

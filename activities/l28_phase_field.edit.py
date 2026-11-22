# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt, time


@app.cell
def _(np):
    # Regular-solution Cahn-Hilliard in reduced units: k_B T = 1, mobility = 1,
    # grid spacing h = 1. Adapted from the MATE 664 EX04 notebook (after
    # T. Yamanaka, PCoMS 2019) with constant mobility.

    def g_prime(c, H):
        return H * (1 - 2 * c) + np.log(c / (1 - c))

    def laplacian(u):
        return (
            np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
        )

    def free_energy(c, H, kappa):
        g = H * c * (1 - c) + c * np.log(c) + (1 - c) * np.log(1 - c)
        gx = np.roll(c, -1, 0) - c
        gy = np.roll(c, -1, 1) - c
        return np.sum(g + 0.5 * kappa * (gx**2 + gy**2))

    def explicit_step(c, H, kappa, dt):
        """Forward Euler on the finite-difference ODEs."""
        mu = g_prime(c, H) - kappa * laplacian(c)
        return np.clip(c + dt * laplacian(mu), 1e-9, 1 - 1e-9)

    def semi_implicit_step(c, H, kappa, dt, A=2.0):
        """Semi-implicit Fourier step: the stiff linear terms are treated
        implicitly, the logarithmic part of g' explicitly."""
        n = c.shape[0]
        k = 2 * np.pi * np.fft.fftfreq(n)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        lap_symbol = (2 - 2 * np.cos(kx)) + (2 - 2 * np.cos(ky))  # same stencil as laplacian()
        c_hat = np.fft.fft2(c)
        rhs = np.fft.fft2(g_prime(c, H) - A * c)
        c_hat = (c_hat - dt * lap_symbol * rhs) / (1 + dt * lap_symbol * (A + kappa * lap_symbol))
        return np.clip(np.real(np.fft.ifft2(c_hat)), 1e-9, 1 - 1e-9)

    def run(method, H, dt, t_end=60.0, n=128, kappa=1.0, seed=374):
        rng = np.random.default_rng(seed)
        c = 0.5 + 0.01 * rng.standard_normal((n, n))
        c += 0.5 - c.mean()
        step = explicit_step if method == "explicit (forward Euler)" else semi_implicit_step
        n_steps = int(round(t_end / dt))
        save = max(1, n_steps // 30)
        times, energies = [0.0], [free_energy(c, H, kappa)]
        for i in range(1, n_steps + 1):
            c = step(c, H, kappa, dt)
            if i % save == 0:
                times.append(i * dt)
                energies.append(free_energy(c, H, kappa))
        return c, np.array(times), np.array(energies), n_steps

    return (run,)


@app.cell(hide_code=True)
def _(mo):
    method = mo.ui.dropdown(
        options=["explicit (forward Euler)", "semi-implicit Fourier"],
        value="explicit (forward Euler)",
        label="time stepping",
    )
    dt_choice = mo.ui.dropdown(
        options=["0.01", "0.02", "0.03", "0.1", "0.5", "1.0"], value="0.02", label="Δt"
    )
    H_slider = mo.ui.slider(
        steps=[1.5, 2.25, 2.5, 3.0], value=2.5, label="H = Ω / kT", show_value=True
    )
    h_nm = mo.ui.dropdown(options=["0.5", "1", "2"], value="1", label="grid spacing h (nm)")
    return H_slider, dt_choice, h_nm, method


@app.cell(hide_code=True)
def _(H_slider, dt_choice, h_nm, method, mo, np, plt, run, time):
    t0 = time.perf_counter()
    c, times, energies, n_steps = run(method.value, H_slider.value, float(dt_choice.value))
    elapsed = time.perf_counter() - t0

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.6, 3.9), layout="constrained", gridspec_kw={"width_ratios": [1, 1.3]}
    )
    im = ax1.imshow(c, cmap="RdBu_r", vmin=0, vmax=1)
    ax1.set(xticks=[], yticks=[], title="composition at t = 60")
    fig.colorbar(im, ax=ax1, label="fraction of B, c", shrink=0.85)
    ax2.plot(times, energies - energies[0], "o-", ms=3, color="#231f20")
    ax2.set(xlabel="time (reduced units)", ylabel="F(t) − F(0)  (units of kT)")

    stable = np.all(np.diff(energies) <= 1e-6) and c.min() > 1e-6 and c.max() < 1 - 1e-6
    verdict = "free energy decreased at every saved step" if stable else "**unstable: free energy rose or c hit 0 or 1**"
    # Real units: without the bond term, the reduced equation at c = 0.5 is ordinary
    # diffusion with D* = g''(0.5) = 4 grid spacings^2 per time unit. Matching D* to
    # the Cu self-diffusion coefficient at 1000 K makes one time unit 4 h^2 / D.
    _D = 2.33e-15
    _h = float(h_nm.value) * 1e-9
    _tau = 4 * _h**2 / _D
    mo.vstack(
        [
            mo.hstack([method, dt_choice, H_slider, h_nm], justify="start", wrap=True),
            mo.md(
                f"Real units with *D* = 2.3 × 10⁻¹⁵ m²/s (Cu at 1000 K): the grid is "
                f"**{128 * float(h_nm.value):.0f} nm** wide, one time unit is **{_tau * 1e3:.1f} ms**, "
                f"and *t* = 60 is **{60 * _tau:.2f} s**."
            ),
            mo.md(
                f"{n_steps:,} steps in **{elapsed:.2f} s**. Mean composition **{c.mean():.5f}**. "
                f"Range **{c.min():.3f} to {c.max():.3f}**. {verdict}."
            ),
            fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()

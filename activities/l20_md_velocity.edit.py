# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import integrate, optimize

    return integrate, mo, np, optimize, plt


@app.cell
def _(np, optimize):
    # Argon in Lennard-Jones units: length sigma, energy epsilon, time tau = sigma * sqrt(m / epsilon)
    SIGMA = 3.405        # Å
    TAU = 2.156          # ps
    T_REDUCED = 20.0 / 119.8   # 20 K, with epsilon / k_B = 119.8 K


    def energy_forces(pos):
        d = pos[:, None, :] - pos[None, :, :]
        r2 = np.sum(d**2, axis=-1)
        np.fill_diagonal(r2, np.inf)
        inv6 = r2**-3
        energy = 2 * np.sum(inv6**2 - inv6)                  # 4 (r^-12 - r^-6), each pair counted twice
        fmag = 24 * (2 * inv6**2 - inv6) / r2
        return energy, np.sum(fmag[:, :, None] * d, axis=1)


    # Start from an icosahedral 13-atom cluster and relax it (Unit 3)
    _g = (1 + 5**0.5) / 2
    _v = [p for s in (-1, 1) for t in (-_g, _g) for p in ((0, s, t), (s, t, 0), (t, 0, s))]
    _start = np.vstack([[0, 0, 0], _v]) * 1.1 / np.sqrt(1 + _g**2)
    _res = optimize.minimize(
        lambda q: energy_forces(q.reshape(-1, 3))[0],
        _start.ravel(),
        jac=lambda q: -energy_forces(q.reshape(-1, 3))[1].ravel(),
        method="L-BFGS-B",
    )
    pos = _res.x.reshape(-1, 3)

    # Random thermal velocities with no net drift of the cluster
    _rng = np.random.default_rng(374)
    vel = _rng.normal(0.0, np.sqrt(T_REDUCED), pos.shape)
    vel -= vel.mean(axis=0)

    # Molecular dynamics: 4000 velocity-Verlet steps of 5 fs (L21 explains this update)
    DT_FS = 5.0
    dt = DT_FS / 1000 / TAU
    n_steps = 4000
    _, F = energy_forces(pos)
    X = np.empty((n_steps + 1, 13)); V = np.empty((n_steps + 1, 13))
    X[0], V[0] = pos[:, 0], vel[:, 0]
    for _k in range(n_steps):
        vel = vel + 0.5 * dt * F
        pos = pos + dt * vel
        _, F = energy_forces(pos)
        vel = vel + 0.5 * dt * F
        X[_k + 1], V[_k + 1] = pos[:, 0], vel[:, 0]

    # Saved trajectory of atom 1 (a surface atom), x component, in Å, Å/ps, and ps
    t_all = np.arange(n_steps + 1) * DT_FS / 1000
    x_all = X[:, 1] * SIGMA
    v_all = V[:, 1] * SIGMA / TAU
    return DT_FS, t_all, v_all, x_all


@app.cell
def _(mo):
    stride = mo.ui.slider(steps=[1, 2, 5, 10, 20, 50, 100], value=10, label="save a frame every s steps, s =", show_value=True)
    return (stride,)


@app.cell
def _(DT_FS, integrate, mo, np, plt, stride, t_all, v_all, x_all):
    s = stride.value
    t, v, x_saved = t_all[::s], v_all[::s], x_all[::s]       # what the saved file contains

    # Displacement over the whole run: one number from each rule
    disp_true = x_saved[-1] - x_saved[0]
    disp_trap = np.trapezoid(v, t)
    disp_simp = integrate.simpson(v, x=t)

    # Position at every saved time: cumulative integrals
    x_trap = x_saved[0] + integrate.cumulative_trapezoid(v, t, initial=0)
    x_simp = x_saved[0] + integrate.cumulative_simpson(v, x=t, initial=0)

    _table = mo.md(
        "\n".join(
            [
                f"**Saved every {s * DT_FS:g} fs, {t.size} frames**",
                "",
                "| | displacement (Å) | error (Å) | largest error along the path (Å) |",
                "|---|---|---|---|",
                f"| stored positions | {disp_true:.5f} | | |",
                f"| trapezoid | {disp_trap:.5f} | {disp_trap - disp_true:+.1e} | {np.max(np.abs(x_trap - x_saved)):.1e} |",
                f"| Simpson 1/3 | {disp_simp:.5f} | {disp_simp - disp_true:+.1e} | {np.max(np.abs(x_simp - x_saved)):.1e} |",
            ]
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, layout="constrained")
    axes[0].plot(t_all, v_all, color="0.75", lw=1, label="every MD step")
    axes[0].plot(t, v, "o", ms=2.5, color="k", label=f"saved every {s} steps")
    axes[0].set(ylabel="vₓ (Å/ps)")
    axes[1].plot(t_all, x_all, color="0.75", lw=2.5, label="stored position")
    axes[1].plot(t, x_trap, color="C3", lw=1.2, label="∫ v dt, trapezoid")
    axes[1].plot(t, x_simp, "--", color="C2", lw=1.2, label="∫ v dt, Simpson 1/3")
    axes[1].set(xlabel="time (ps)", ylabel="x (Å)")
    for _ax in axes:
        _ax.grid(alpha=0.25)
        _ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)

    mo.vstack([stride, _table, fig])
    return


if __name__ == "__main__":
    app.run()

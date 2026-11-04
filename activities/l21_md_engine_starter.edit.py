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
    from scipy import optimize

    return mo, np, optimize, plt


@app.cell
def _(mo):
    mo.md("""
    ## Build the engine: an isolated 13-atom cluster

    Complete the TODO lines in order: the central difference in `forces`, then `step_euler`, then `step_verlet`.
    Each later cell waits until the functions it needs are complete.
    Positions, time, and energy use Lennard–Jones reduced units; every mass is 1.
    The starting atoms are displaced from a relaxed cluster, with zero initial velocity.
    """)
    return


@app.cell
def _(np):
    def energy_func(r):
        # Lennard-Jones pair energy V(r) in reduced units.
        return 4 * (r**-12 - r**-6)

    def internal_energy(pos):
        # Every pair i < j at once: the same sum as a double loop over atoms.
        i, j = np.triu_indices(len(pos), k=1)
        r_ij = np.linalg.norm(pos[i] - pos[j], axis=1)
        return np.sum(energy_func(r_ij))

    return (internal_energy,)


@app.cell
def _(internal_energy, np):
    def forces(pos, delta=1e-5):
        F = np.zeros_like(pos)
        for i in range(len(pos)):  # atom
            for k in range(3):  # direction: 0, 1, 2 for x, y, z
                pos_r = pos.copy()
                pos_r[i, k] += delta
                pos_l = pos.copy()
                pos_l[i, k] -= delta
                F[i, k] = np.nan  # TODO: central difference of internal_energy(pos_r) and internal_energy(pos_l)
        return F

    return (forces,)


@app.cell
def _(forces, mo, np):
    forces_ready = bool(np.all(np.isfinite(forces(np.array([[0.0, 0, 0], [1.2, 0, 0]])))))
    mo.stop(not forces_ready, mo.md("**Complete the central difference in `forces`.** The minimization check runs once it returns finite values."))
    return (forces_ready,)


@app.cell
def _(forces, forces_ready, internal_energy, mo, np, optimize):
    mo.stop(not forces_ready)
    # Noisy icosahedral guess, relaxed with the numerical forces as the negative gradient.
    phi = (1 + np.sqrt(5)) / 2
    shell = []
    for a in [-1, 1]:
        for b in [-phi, phi]:
            shell.extend([[0, a, b], [a, b, 0], [b, 0, a]])
    base = np.vstack([np.zeros((1, 3)), np.array(shell)])
    pos_guess = 0.6 * base + 0.05 * np.random.default_rng(21).normal(size=base.shape)
    result = optimize.minimize(
        lambda x: internal_energy(x.reshape(-1, 3)),
        pos_guess.ravel(),
        jac=lambda x: -forces(x.reshape(-1, 3)).ravel(),
        method="BFGS",
        options={"gtol": 1e-7},
    )
    pos_min = result.x.reshape(-1, 3)
    pos_min = pos_min - pos_min.mean(axis=0)
    U_min = internal_energy(pos_min)
    F_min = forces(pos_min)
    mo.md(
        rf"""
    **Minimization check**

    | Quantity | Value |
    |---|---:|
    | Energy of the guess | {internal_energy(pos_guess):.5f} |
    | Relaxed energy $U_{{\min}}$ | {U_min:.6f} |
    | Known icosahedral minimum | −44.326801 |
    | Largest force component at the minimum | {np.max(np.abs(F_min)):.1e} |
    | Largest component of the summed force | {np.max(np.abs(F_min.sum(axis=0))):.1e} |
    """
    )
    return U_min, pos_min


@app.cell
def _(U_min, internal_energy, np, optimize, pos_min):
    def initial_state(added_energy):
        direction = np.random.default_rng(374).normal(size=pos_min.shape)
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        target = U_min + len(pos_min) * added_energy
        amplitude = optimize.brentq(
            lambda s: internal_energy(pos_min + s * direction) - target, 0, 0.8
        )
        return pos_min + amplitude * direction, np.zeros_like(pos_min)

    return (initial_state,)


@app.cell
def _():
    def step_euler(pos, vel, F, dt, m=1.0):
        return None  # TODO: return pos_next, vel_next, both from the current pos and vel

    def step_verlet(pos, pos_prev, F, dt, m=1.0):
        return None  # TODO: return pos_next from pos, pos_prev, and F

    return step_euler, step_verlet


@app.cell
def _(forces, internal_energy, np, step_euler, step_verlet):
    def run(pos, vel, dt, duration, method):
        pos, vel = pos.copy(), vel.copy()
        F = forces(pos)
        # Taylor expansion supplies the missing position at t = -dt.
        pos_prev = pos - dt * vel + 0.5 * dt**2 * F
        times, frames, internal, kinetic = [], [], [], []
        stopped = ""
        n_steps = round(duration / dt)
        for n in range(n_steps + 1):
            U = internal_energy(pos)
            F = forces(pos)
            if method == "Størmer–Verlet":
                pos_next = step_verlet(pos, pos_prev, F, dt)
                vel = (pos_next - pos_prev) / (2 * dt)  # velocity at the current time
            else:
                pos_next, vel_next = step_euler(pos, vel, F, dt)
            K = 0.5 * np.sum(vel**2)
            if not np.isfinite(U + K) or np.max(np.abs(pos)) > 1e5:
                stopped = "Numerical overflow or positions beyond the display safety limit."
                break
            times.append(n * dt)
            frames.append(pos.copy())
            internal.append(U)
            kinetic.append(K)
            if n == n_steps:
                break
            pos_prev, pos = pos, pos_next
            if method == "forward Euler":
                vel = vel_next
        return {
            "t": np.array(times), "x": np.array(frames),
            "U": np.array(internal), "K": np.array(kinetic),
            "stopped": stopped,
        }

    return (run,)


@app.cell
def _(mo):
    method = mo.ui.dropdown(
        ["forward Euler", "Størmer–Verlet"], value="forward Euler", label="method"
    )
    dt = mo.ui.dropdown(
        options={"0.002": 0.002, "0.005": 0.005, "0.02": 0.02,
                 "0.05": 0.05, "0.08": 0.08},
        value="0.005", label="time step"
    )
    added = mo.ui.slider(steps=[0.1, 0.3, 0.6, 1.0], value=0.3,
                         label="added energy per atom", show_value=True)
    mo.hstack([method, dt, added], justify="start", gap=2)
    return added, dt, method


@app.cell
def _(added, dt, initial_state, method, mo, np, run, step_euler, step_verlet):
    probe = np.zeros((2, 3))
    step = step_euler(probe, probe, probe, 0.01) if method.value == "forward Euler" else step_verlet(probe, probe, probe, 0.01)
    name = "step_euler" if method.value == "forward Euler" else "step_verlet"
    mo.stop(step is None, mo.md(f"**Complete `{name}`** to run {method.value}."))
    start, velocity = initial_state(added.value)
    trajectory = run(start, velocity, dt.value, 10.0, method.value)
    return (trajectory,)


@app.cell
def _(mo, trajectory):
    frame = mo.ui.slider(0, len(trajectory["t"]) - 1, value=0,
                         label="trajectory frame", show_value=True)
    frame
    return (frame,)


@app.cell
def _(frame, method, mo, np, plt, trajectory):
    t, xyz = trajectory["t"], trajectory["x"]
    U, K = trajectory["U"], trajectory["K"]
    E = U + K
    fig = plt.figure(figsize=(10, 4.2), layout="constrained")
    ax = fig.add_subplot(121, projection="3d")
    p = xyz[frame.value]
    visible = np.all(np.abs(p) <= 4, axis=1)
    ax.scatter(p[visible, 0], p[visible, 1], p[visible, 2], s=180,
               color="#b5473a", edgecolors="black")
    ax.set(xlim=(-4, 4), ylim=(-4, 4), zlim=(-4, 4),
           xlabel="x", ylabel="y", zlabel="z", title=f"t = {t[frame.value]:.3f}")
    ax.set_box_aspect((1, 1, 1))
    outside = np.count_nonzero(np.any(np.abs(p) > 4, axis=1))
    ax.text2D(0.02, 0.02, f"Atoms outside view: {outside}", transform=ax.transAxes)
    energy_ax = fig.add_subplot(122)
    energy_ax.plot(t, K, label="kinetic K", color="C3")
    energy_ax.plot(t, U - U[0], label="internal U − U(0)", color="C0")
    energy_ax.plot(t, E - E[0], label="total E − E(0)", color="black")
    energy_ax.set(xlabel="time", ylabel="energy change / kinetic energy",
                  title=method.value)
    energy_ax.grid(alpha=0.2)
    energy_ax.legend(fontsize=10)
    drift_fig, drift_ax = plt.subplots(figsize=(9, 2.6), layout="constrained")
    drift_ax.plot(t, E - E[0], color="black")
    drift_ax.set(xlabel="time", ylabel="E(t) − E(0)", title="Total-energy error")
    drift_ax.grid(alpha=0.2)
    message = (
        f"**Maximum |E(t) − E(0)|:** {np.max(np.abs(E-E[0])):.3g}. "
        f"**Initial total energy:** {E[0]:.5f}. "
        f"**Final time:** {t[-1]:g}."
    )
    mo.vstack([mo.md(message), fig, drift_fig, mo.md(trajectory["stopped"])])
    return


if __name__ == "__main__":
    app.run()

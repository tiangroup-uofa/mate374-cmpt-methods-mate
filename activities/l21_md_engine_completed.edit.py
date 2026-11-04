# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12", "matplotlib>=3.9", "ase>=3.26", "weas-widget>=0.2.6"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import optimize
    from ase import Atoms
    from weas_widget import WeasWidget

    return Atoms, WeasWidget, mo, np, optimize, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From relaxation to vibration: an isolated cluster

    **Prediction:** after we relax the cluster and displace its atoms, will its
    total energy stay constant? Compare the methods at the same time step.

    - Length, time, and energy use Lennard–Jones reduced units. Every mass is 1.
    - Every pair interacts, with no cutoff, container, or heat bath.
    - First, minimize the energy and watch the forces decrease. Then release a
      displaced cluster with zero initial velocity and follow its motion.
    """)
    return


@app.cell
def _(np):
    def energy_func(r):
        # Lennard-Jones pair energy V(r) in reduced units.
        return 4 * (r**-12 - r**-6)

    def U(pos):
        # Every pair i < j at once: the same sum as a double loop over atoms.
        i, j = np.triu_indices(len(pos), k=1)
        r_ij = np.linalg.norm(pos[i] - pos[j], axis=1)
        return np.sum(energy_func(r_ij))

    return U, energy_func


@app.cell
def _(U, np):
    def forces(pos):
        delta = 1e-5  # coordinate spacing, independent of the time step
        F = np.zeros_like(pos)
        for i in range(len(pos)):  # atom
            for k in range(3):  # direction: 0, 1, 2 for x, y, z
                pos_r = pos.copy()
                pos_r[i, k] += delta
                pos_l = pos.copy()
                pos_l[i, k] -= delta
                F[i, k] = -(U(pos_r) - U(pos_l)) / (2 * delta)
        return F

    return (forces,)


@app.cell(hide_code=True)
def _(Atoms, WeasWidget, mo):
    def cluster_viewer(positions):
        # Supplied visualization: WEAS plays positions, not the MD calculation.
        # Convert reduced lengths to argon Angstroms for sensible atom radii.
        frames = [Atoms("Ar" * len(p), positions=3.4 * p, pbc=False) for p in positions]
        gui = {
            "controls": {"enabled": False},
            "meshLegend": {"enabled": False},
            "atomLegend": {"enabled": False},
            "buttons": {"enabled": True, "fullscreen": True, "undo": False,
                        "redo": False, "export": False, "import": False,
                        "measurement": False},
            "timeline": {"enabled": True},
        }
        viewer = WeasWidget(
            from_ase=frames,
            viewerStyle={"width": "100%", "height": "340px"},
            guiConfig=gui,
            modelStyle=0,
            cellSettings={"showCell": False},
            cameraSetting={"lookAt": [0, 0, 0], "direction": [1, 0.6, 0.4],
                           "distance": 32, "zoom": 1},
            animationState={"isPlaying": False, "frameDuration": 100},
        )
        # WEAS wraps its AnyWidget canvas in an ipywidgets HBox.
        return mo.ui.anywidget(viewer.children[0])

    return (cluster_viewer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 · Relaxation with the energy and its gradient

    BFGS minimizes `U(pos)`. Its gradient is `-forces(pos)`.
    The progress slider follows optimizer iterations, which have no physical time.
    Use the play button or drag the slider to compare the energy and largest force.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    settings = mo.ui.dictionary({
        "atoms": mo.ui.number(start=2, stop=13, step=1, value=13, label="number of atoms"),
        "steps": mo.ui.number(start=10, stop=5000, step=10, value=1000, label="total steps"),
        "method": mo.ui.dropdown(["forward Euler", "Størmer–Verlet"],
                                 value="Størmer–Verlet", label="method"),
        "dt": mo.ui.dropdown({"0.002": 0.002, "0.005": 0.005, "0.02": 0.02,
                              "0.04": 0.04, "0.08": 0.08},
                             value="0.005", label="time step"),
        "added": mo.ui.slider(steps=[0.1, 0.3, 0.6, 1.0], value=0.3,
                              label="added energy per atom", show_value=True),
    }).form(submit_button_label="Simulate", show_clear_button=False)
    mo.vstack([mo.md("Choose the settings, then click **Simulate**. Editing a control does not start a run. "
                     "Browser limits: 2–13 atoms and 10–5,000 steps."), settings])
    return (settings,)


@app.cell(hide_code=True)
def _(mo, settings):
    mo.stop(settings.value is None, mo.md("Click **Simulate** to relax the cluster and run the dynamics."))
    config = settings.value
    return (config,)


@app.cell
def _(U, config, forces, np, optimize):
    # Noisy icosahedral guess, relaxed with the negative force as the gradient.
    phi = (1 + np.sqrt(5)) / 2
    shell = []
    for a in [-1, 1]:
        for b in [-phi, phi]:
            shell.extend([[0, a, b], [a, b, 0], [b, 0, a]])
    base = np.vstack([np.zeros((1, 3)), np.array(shell)])[:int(config["atoms"])]
    pos_guess = 0.6 * base + 0.05 * np.random.default_rng(21).normal(size=base.shape)
    pos_guess -= pos_guess.mean(axis=0)
    relaxation_frames = [pos_guess.copy()]
    result = optimize.minimize(
        lambda x: U(x.reshape(-1, 3)),
        pos_guess.ravel(),
        jac=lambda x: -forces(x.reshape(-1, 3)).ravel(),
        callback=lambda x: relaxation_frames.append(x.reshape(-1, 3).copy()),
        method="BFGS",
        options={"gtol": 1e-7, "maxiter": 300},
    )
    pos_min = result.x.reshape(-1, 3)
    U_min = U(pos_min)
    F_min = forces(pos_min)
    return F_min, U_min, pos_min, relaxation_frames, result


@app.cell(hide_code=True)
def _(U, cluster_viewer, forces, np, relaxation_frames):
    relaxation_energy = np.array([U(p) for p in relaxation_frames])
    relaxation_force = np.array([
        np.max(np.linalg.norm(forces(p), axis=1)) for p in relaxation_frames
    ])
    relaxation_view = cluster_viewer(relaxation_frames)
    relaxation_view
    return relaxation_energy, relaxation_force, relaxation_view


@app.cell(hide_code=True)
def _(mo, relaxation_energy, relaxation_force, relaxation_view, result):
    _i = min(int(relaxation_view.value.get("currentFrame", 0)), len(relaxation_energy) - 1)
    _warning = "" if result.success else f"**Relaxation stopped:** {result.message}"
    mo.vstack([mo.md(rf"""
    **Energy:** $U = {relaxation_energy[_i]:.6f}$ ·
    **Largest force:** $\max_i\lVert\mathbf F_i\rVert = {relaxation_force[_i]:.3g}$
    """), mo.md(_warning)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2 · Release a displaced cluster

    The same seeded displacement direction sets the starting shape for both methods.
    Its amplitude gives $U(0)=U_{\min}+N\,\Delta e$, where the slider sets $\Delta e$.
    All initial velocities are zero. The forces then accelerate the atoms.

    Both step functions take `(pos, vel, pos_prev, F, dt, m=1)` and return
    `(pos_next, vel_next)`. Euler ignores `pos_prev`. Position Verlet returns
    `None` for `vel_next` because its position update needs no velocity.
    """)
    return


@app.cell(hide_code=True)
def _(U, U_min, np, optimize, pos_min):
    def initial_state(added_energy):
        direction = np.random.default_rng(374).normal(size=pos_min.shape)
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        target = U_min + len(pos_min) * added_energy
        # Bracket the first energy crossing along the displacement direction.
        residual = lambda s: U(pos_min + s * direction) - target
        upper = 0.01
        while residual(upper) < 0 and upper < 100:
            upper *= 1.2
        if residual(upper) < 0:
            # Compression can supply energies above the dissociation limit too.
            direction = -pos_min / np.linalg.norm(pos_min)
            upper = 0.99 * np.linalg.norm(pos_min)
        amplitude = optimize.brentq(residual, 0, upper)
        return pos_min + amplitude * direction, np.zeros_like(pos_min)

    return (initial_state,)


@app.cell
def _():
    def step_euler(pos, vel, pos_prev, F, dt, m=1):
        pos_next = pos + vel * dt
        vel_next = vel + F / m * dt
        return pos_next, vel_next

    def step_verlet(pos, vel, pos_prev, F, dt, m=1):
        pos_next = 2 * pos - pos_prev + F / m * dt**2
        return pos_next, None

    return step_euler, step_verlet


@app.cell
def _(forces, np):
    def run(pos, vel, dt, n_steps, step, m=1):
        pos, vel = pos.copy(), vel.copy()
        # Taylor expansion supplies the missing position at t = -dt.
        pos_prev = pos - dt * vel + 0.5 * dt**2 * forces(pos) / m
        positions = [pos.copy()]
        for n in range(n_steps):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                F = forces(pos)
                pos_next, vel_next = step(pos, vel, pos_prev, F, dt, m)
            # Stop a divergent run before it sends invalid coordinates to WEAS.
            if not np.all(np.isfinite(pos_next)) or np.max(np.abs(pos_next)) > 1e5:
                break
            positions.append(pos_next.copy())
            pos_prev, pos, vel = pos, pos_next, vel_next
        return np.array(positions)

    return (run,)


@app.cell(hide_code=True)
def _(U, np):
    def analyse(positions, dt, m=1):
        # Supplied analysis: use identical central differences for BOTH methods.
        # Each velocity and energy belongs to its matching interior position.
        velocities = (positions[2:] - positions[:-2]) / (2 * dt)
        xyz = positions[1:-1]
        times = np.arange(1, len(positions) - 1) * dt
        internal = np.array([U(p) for p in xyz])
        kinetic = 0.5 * m * np.sum(velocities**2, axis=(1, 2))
        return {"t": times, "x": xyz, "v": velocities,
                "U": internal, "K": kinetic, "E": internal + kinetic}

    return (analyse,)


@app.cell(hide_code=True)
def _(analyse, config, initial_state, mo, run, step_euler, step_verlet):
    start, velocity = initial_state(config["added"])
    step = {"forward Euler": step_euler, "Størmer–Verlet": step_verlet}[config["method"]]
    positions = run(start, velocity, config["dt"], int(config["steps"]), step)
    mo.stop(len(positions) < 3, mo.md("Fewer than 3 positions survived. Reduce the time step."))
    trajectory = analyse(positions, config["dt"])
    stopped = len(positions) < int(config["steps"]) + 1
    return positions, stopped, trajectory


@app.cell(hide_code=True)
def _(cluster_viewer, np, trajectory):
    # Limit the browser animation to 201 frames. Analysis keeps every MD step.
    display_indices = np.unique(np.linspace(0, len(trajectory["t"]) - 1,
                                             min(201, len(trajectory["t"]))).astype(int))
    dynamics_view = cluster_viewer(trajectory["x"][display_indices])
    dynamics_view
    return display_indices, dynamics_view


@app.cell(hide_code=True)
def _(display_indices, dynamics_view, mo, np, trajectory):
    _frame = min(int(dynamics_view.value.get("currentFrame", 0)), len(display_indices) - 1)
    _i = display_indices[_frame]
    _radius = np.linalg.norm(trajectory["x"][_i], axis=1).max()
    mo.md(
        f"**t = {trajectory['t'][_i]:.3f}** · "
        f"**U = {trajectory['U'][_i]:.5f}** · "
        f"**K = {trajectory['K'][_i]:.5f}** · "
        f"**E = {trajectory['E'][_i]:.5f}** · "
        f"largest radius = {_radius:.2f}."
    )
    return


@app.cell(hide_code=True)
def _(config, mo, np, plt, stopped, trajectory):
    _t = trajectory["t"]
    _total = trajectory["E"]
    _fig, _ax = plt.subplots(figsize=(9, 3.7), layout="constrained")
    _ax.plot(_t, _total - _total[0], color="black")
    _ax.set(xlabel="time", ylabel="E(t) − E(t₁)",
            title=f"Total-energy change · {config['method']} · {config['atoms']} atoms")
    _ax.grid(alpha=0.2)
    _message = (
        f"**Maximum |E(t) − E(t₁)|:** {np.max(np.abs(_total - _total[0])):.3g}. "
        f"**First analysed time t₁:** {_t[0]:g}. **Last analysed time:** {_t[-1]:g}."
    )
    _warning = ("**Run stopped at the coordinate safety limit. Reduce the time step.**"
                if stopped else "")
    mo.vstack([mo.md(_message), _fig, mo.md(_warning), mo.md(r"""
    Central differences need positions on both sides, so the energy curves omit
    the first and last stored positions. $U$, $K$, and $E$ use the same interior times.
    The animation samples at most 201 frames, while the curves use every interior step.

    **Compare:** keep the added energy fixed, select Euler, then halve the time step.
    Repeat with Verlet. Physical vibration exchanges internal and kinetic energy.
    Numerical drift changes their sum. Raising the starting energy can increase
    motion or cause atoms to escape, so compare time steps before interpreting a rearrangement.
    Energy conservation is one check. Trajectory accuracy also requires time-step refinement.
    """)])
    return


if __name__ == "__main__":
    app.run()

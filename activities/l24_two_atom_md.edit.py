# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.0", "numpy>=2.0", "scipy>=1.12", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq

    return brentq, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## Two atoms, one bound trajectory

    How do starting conditions and the integrator change the motion and energy
    of an isolated equal-mass pair?
    """)
    return


@app.cell
def _(mo):
    added_energy = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.20,
        label="Added energy Eadd (LJ energy units)", show_value=True,
    )
    preparation = mo.ui.dropdown(
        ["Velocity kick at minimum", "Compressed turning point",
         "Stretched turning point"],
        value="Velocity kick at minimum",
        label="Initial preparation",
    )
    integrator = mo.ui.dropdown(
        ["Forward Euler", "Position Verlet", "Velocity Verlet"],
        value="Velocity Verlet",
        label="Time integrator",
    )
    time_step = mo.ui.dropdown(
        options={"0.002": 0.002, "0.005": 0.005, "0.01": 0.01,
                 "0.02": 0.02, "0.04": 0.04},
        value="0.005",
        label="Time step dt (LJ time units)",
    )
    mo.vstack([added_energy, preparation, integrator, time_step])
    return added_energy, integrator, preparation, time_step


@app.cell
def _(mo):
    mo.md(r"""
    **LJ reduced units:** \(m=\sigma=\epsilon=1\), \(x_1=-r/2\), \(x_2=r/2\).
    The pair energy is \(U(r)=4(r^{-12}-r^{-6})\), with its minimum
    \(U(2^{1/6})=-1\). Thus \(0<E_{\mathrm{add}}<1\) gives the bound target
    \(E_0=-1+E_{\mathrm{add}}<0\). The compressed and stretched starts solve
    \(U(r)=E_0\) with brentq.
    """)
    return


@app.cell
def _(np):
    def energy_func(r):
        # Lennard–Jones pair energy in reduced units.
        return 4 * (r**-12 - r**-6)

    def U(pos):
        # Sum each unique pair i < j, as in the L21 MD engine.
        atom_i, atom_j = np.triu_indices(len(pos), k=1)
        separations = np.linalg.norm(pos[atom_i] - pos[atom_j], axis=1)
        return np.sum(energy_func(separations))

    def forces(pos):
        # Central finite differences of U; this spacing is independent of dt.
        delta = 1e-5
        force = np.zeros_like(pos)
        for atom in range(len(pos)):
            for direction in range(pos.shape[1]):
                pos_right = pos.copy()
                pos_right[atom, direction] += delta
                pos_left = pos.copy()
                pos_left[atom, direction] -= delta
                force[atom, direction] = -(U(pos_right) - U(pos_left)) / (2 * delta)
        return force

    return U, energy_func, forces


@app.cell
def _(brentq, energy_func, np):
    def initial_state(energy_added, preparation_name):
        r_minimum = 2 ** (1 / 6)
        energy_target = energy_func(r_minimum) + energy_added

        if preparation_name == "Velocity kick at minimum":
            separation = r_minimum
            speed = np.sqrt(energy_added)
            velocities = np.array([[-speed], [speed]])
            root_residual = None
        else:
            residual = lambda r: energy_func(r) - energy_target
            if preparation_name == "Compressed turning point":
                separation = brentq(residual, 0.70, r_minimum)
            else:
                upper = 2 * r_minimum
                while residual(upper) < 0:
                    upper *= 2
                separation = brentq(residual, r_minimum, upper)
            velocities = np.zeros((2, 1))
            root_residual = abs(residual(separation))

        positions = np.array([[-separation / 2], [separation / 2]])
        initial_energy = energy_func(separation) + 0.5 * np.sum(velocities**2)
        energy_residual = abs(initial_energy - energy_target)
        return positions, velocities, separation, energy_target, root_residual, energy_residual

    return (initial_state,)


@app.cell
def _(forces):
    def euler_step(pos, vel, force, dt, mass=1):
        pos_next = pos + vel * dt
        vel_next = vel + force / mass * dt
        return pos_next, vel_next

    def position_verlet_step(pos, pos_previous, force, dt, mass=1):
        pos_next = 2 * pos - pos_previous + force / mass * dt**2
        return pos_next

    def velocity_verlet_step(pos, vel, force, dt, mass=1):
        # Half-kick, drift, half-kick. Return force_next for the following step.
        vel_half = vel + 0.5 * dt * force / mass
        pos_next = pos + dt * vel_half
        force_next = forces(pos_next)
        vel_next = vel_half + 0.5 * dt * force_next / mass
        return pos_next, vel_next, force_next

    return euler_step, position_verlet_step, velocity_verlet_step


@app.cell
def _(euler_step, forces, np, position_verlet_step, velocity_verlet_step):
    def integrate(pos0, vel0, dt, duration, method):
        n_steps = int(round(duration / dt))
        is_position_verlet = method == "Position Verlet"
        # Position Verlet needs one extra endpoint to centre the final velocity.
        steps_to_compute = n_steps + int(is_position_verlet)
        positions = np.empty((steps_to_compute + 1, 2, 1))
        velocities = np.empty_like(positions)
        positions[0] = pos0
        velocities[0] = vel0

        force = forces(pos0)
        pos_previous = pos0 - dt * vel0 + 0.5 * dt**2 * force
        position = pos0.copy()
        velocity = vel0.copy()
        completed = 0
        stopped = False

        for step_index in range(steps_to_compute):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if method == "Forward Euler":
                    pos_next, vel_next = euler_step(position, velocity, force, dt)
                    force_next = forces(pos_next)
                elif is_position_verlet:
                    pos_next = position_verlet_step(
                        position, pos_previous, force, dt
                    )
                    vel_next = None
                    force_next = forces(pos_next)
                else:
                    pos_next, vel_next, force_next = velocity_verlet_step(
                        position, velocity, force, dt
                    )

            next_separation = pos_next[1, 0] - pos_next[0, 0]
            safe = (np.all(np.isfinite(pos_next))
                    and np.max(np.abs(pos_next)) < 100
                    and next_separation > 0.65)
            if not safe:
                stopped = True
                break

            positions[step_index + 1] = pos_next
            if vel_next is not None:
                velocities[step_index + 1] = vel_next
            pos_previous, position = position, pos_next
            if vel_next is not None:
                velocity = vel_next
            force = force_next
            completed += 1

        if is_position_verlet:
            if stopped:
                output_frames = max(1, completed)
            else:
                output_frames = n_steps + 1
            computed_positions = positions[:completed + 1]
            output_positions = computed_positions[:output_frames]
            output_velocities = np.empty_like(output_positions)
            output_velocities[0] = vel0
            for frame_index in range(1, output_frames):
                output_velocities[frame_index] = (
                    computed_positions[frame_index + 1]
                    - computed_positions[frame_index - 1]
                ) / (2 * dt)
        else:
            output_positions = positions[:completed + 1]
            output_velocities = velocities[:completed + 1]

        return output_positions, output_velocities, stopped

    return (integrate,)


@app.cell
def _(added_energy, initial_state, integrator, integrate, preparation, time_step):
    (start_positions, start_velocities, start_separation, target_energy,
     root_residual, start_energy_residual) = initial_state(
        added_energy.value, preparation.value
    )
    dt = time_step.value
    duration = 12.0
    positions, velocities, stopped = integrate(
        start_positions, start_velocities, dt, duration, integrator.value
    )
    return (dt, duration, positions, root_residual, start_energy_residual,
            start_separation, start_velocities, start_positions, stopped,
            target_energy, velocities)


@app.cell
def _(U, np, positions, velocities, dt):
    times = np.arange(len(positions)) * dt
    separations = positions[:, 1, 0] - positions[:, 0, 0]
    potential_energy = np.array([U(pos) for pos in positions])
    kinetic_energy = 0.5 * np.sum(velocities**2, axis=(1, 2))
    total_energy = potential_energy + kinetic_energy
    momentum = np.sum(velocities, axis=1)
    max_energy_drift = np.max(np.abs(total_energy - total_energy[0]))
    max_momentum = np.max(np.abs(momentum))
    return (kinetic_energy, max_energy_drift, max_momentum,
            potential_energy, separations, times, total_energy)


@app.cell(hide_code=True)
def _(energy_func, forces, initial_state, integrate, np):
    def run_numerical_checks():
        # Check both turning roots at the page's worked-example energy.
        compressed = initial_state(0.20, "Compressed turning point")
        stretched = initial_state(0.20, "Stretched turning point")
        assert abs(compressed[2] - 1.0553979) < 1e-6
        assert abs(stretched[2] - 1.2390213) < 1e-6
        assert compressed[4] < 1e-10 and stretched[4] < 1e-10
        assert compressed[5] < 1e-10 and stretched[5] < 1e-10

        check_positions, check_velocities, _, _, _, _ = initial_state(
            0.30, "Velocity kick at minimum"
        )
        force_balance = np.max(np.abs(np.sum(forces(check_positions), axis=0)))
        assert force_balance < 1e-8

        # Compare fixed-time final states with a finer velocity-Verlet run.
        check_duration = 2.0
        check_steps = [0.04, 0.02, 0.01]
        reference_dt = check_steps[0] / 64
        reference_positions, reference_velocities, _ = integrate(
            check_positions, check_velocities, reference_dt,
            check_duration, "Velocity Verlet"
        )
        reference_state = np.concatenate(
            [reference_positions[-1].ravel(), reference_velocities[-1].ravel()]
        )
        errors = []
        for check_dt in check_steps:
            check_pos, check_vel, _ = integrate(
                check_positions, check_velocities, check_dt,
                check_duration, "Velocity Verlet"
            )
            check_state = np.concatenate([check_pos[-1].ravel(), check_vel[-1].ravel()])
            errors.append(np.linalg.norm(check_state - reference_state))
        ratios = np.array(errors[:-1]) / np.array(errors[1:])
        assert 3.2 < ratios[0] < 4.8
        assert 3.2 < ratios[1] < 4.8

        # A bound default velocity-Verlet run should preserve momentum and energy.
        default_pos, default_vel, _, _, _, _ = initial_state(
            0.20, "Velocity kick at minimum"
        )
        path, speeds, stopped = integrate(
            default_pos, default_vel, 0.005, 12.0, "Velocity Verlet"
        )
        assert not stopped
        assert np.max(np.abs(np.sum(speeds, axis=1))) < 1e-8
        energies = np.array([
            energy_func(path_i[1, 0] - path_i[0, 0]) +
            0.5 * np.sum(speed_i**2)
            for path_i, speed_i in zip(path, speeds)
        ])
        assert np.max(np.abs(energies - energies[0])) < 1e-3
        return (compressed[2], stretched[2], force_balance,
                errors, ratios, np.max(np.abs(energies - energies[0])))

    return (run_numerical_checks,)


@app.cell(hide_code=True)
def _(mo, run_numerical_checks):
    if mo.app_meta().mode == "script":
        compressed_root, stretched_root, net_force, errors, ratios, drift = run_numerical_checks()
        print(
            "Checks: turning roots "
            f"{compressed_root:.7f}, {stretched_root:.7f}; "
            f"force balance {net_force:.2e}; "
            f"VV refinement ratios {ratios[0]:.2f}, {ratios[1]:.2f}; "
            f"default energy drift {drift:.2e}."
        )
    return


@app.cell
def _(mo, np, positions):
    # Frame selection redraws the view; it does not rerun the trajectory.
    display_indices = np.unique(
        np.linspace(0, len(positions) - 1, min(241, len(positions))).astype(int)
    )
    frame = mo.ui.slider(
        start=0, stop=max(0, len(display_indices) - 1), step=1, value=0,
        label="Motion frame", show_value=True,
    )
    frame
    return display_indices, frame


@app.cell(hide_code=True)
def _(energy_func, frame, integrator, mo, np, plt, display_indices,
      kinetic_energy, max_energy_drift, max_momentum, potential_energy,
      positions, preparation, root_residual, separations, start_energy_residual,
      start_separation, target_energy, times, total_energy, dt, stopped,
      velocities):
    frame_index = min(frame.value, len(display_indices) - 1)
    sample_index = int(display_indices[frame_index])
    current_time = times[sample_index]
    separation = separations[sample_index]
    current_potential = potential_energy[sample_index]
    current_kinetic = kinetic_energy[sample_index]
    current_total = total_energy[sample_index]

    with plt.rc_context({"font.size": 20, "axes.titlesize": 22,
                         "axes.labelsize": 19, "legend.fontsize": 16}):
        figure = plt.figure(figsize=(8.2, 12.5), layout="constrained")
        grid = figure.add_gridspec(3, 1, height_ratios=[0.9, 1.25, 0.9])
        motion_axis = figure.add_subplot(grid[0, 0])
        potential_axis = figure.add_subplot(grid[1, 0])
        drift_axis = figure.add_subplot(grid[2, 0])

        x1, x2 = positions[sample_index, :, 0]
        scene_extent = max(1.1, 0.65 * float(np.max(separations)) + 0.25)
        motion_axis.plot([x1, x2], [0.20, 0.20], color="0.55", lw=3, zorder=1)
        motion_axis.scatter([x1, x2], [0.20, 0.20], s=720,
                            c=["#2673b8", "#e36c2f"], edgecolor="white",
                            linewidth=2, zorder=2)
        motion_axis.text(x1, 0.67, "Atom 1", ha="center", va="center",
                         fontsize=12, weight="bold")
        motion_axis.text(x2, 0.67, "Atom 2", ha="center", va="center",
                         fontsize=12, weight="bold")
        motion_axis.annotate(
            "", xy=(x2, -0.43), xytext=(x1, -0.43),
            arrowprops={"arrowstyle": "<->", "lw": 2.2, "color": "#333333"},
        )
        motion_axis.text(
            0, -0.78, f"separation r = {separation:.3f} σ",
            ha="center", va="center", fontsize=14, weight="bold",
        )
        motion_axis.set_xlim(-scene_extent, scene_extent)
        motion_axis.set_ylim(-1.05, 0.95)
        motion_axis.set_yticks([])
        motion_axis.set_xlabel("Position along the line, x (σ)")
        motion_axis.tick_params(labelsize=16)
        motion_axis.set_title("Two equal-mass atoms")
        motion_axis.grid(axis="x", alpha=0.2)

        curve_right = max(2.8, 1.12 * float(np.max(separations)))
        r_curve = np.linspace(0.90, curve_right, 600)
        u_curve = energy_func(r_curve)
        potential_axis.plot(r_curve, u_curve, color="#2673b8", lw=2.6,
                            label="LJ potential U(r)")
        potential_axis.axhline(total_energy[0], color="0.25", ls="--", lw=1.8,
                               label="Initial energy E(0)")
        potential_axis.axhline(current_total, color="#c64b40", lw=1.7,
                               label="Current energy E(t)")
        potential_axis.plot(separation, current_potential, "o", ms=10,
                            color="#e36c2f", mec="white", mew=1.4, zorder=4)
        potential_axis.vlines(separation, current_potential, current_total,
                              color="#e36c2f", ls=":", lw=2.4)
        potential_axis.annotate(
            f"K = {current_kinetic:.3f}",
            (separation, 0.5 * (current_potential + current_total)),
            xytext=(8, 2), textcoords="offset points",
            fontsize=11, weight="bold", color="#8d3c1c",
        )
        potential_axis.annotate(
            f"U = {current_potential:.3f}",
            (separation, current_potential),
            xytext=(7, -17), textcoords="offset points",
            fontsize=11, color="#1b5485",
        )
        lower_energy = min(-1.28, float(np.min(potential_energy)) - 0.12)
        upper_energy = max(0.25, float(np.max(total_energy)) + 0.12)
        potential_axis.set_ylim(lower_energy, upper_energy)
        potential_axis.set_xlim(0.90, curve_right)
        potential_axis.set_xlabel("Separation r (σ)")
        potential_axis.set_ylabel("Energy (ε)")
        potential_axis.tick_params(labelsize=16)
        potential_axis.set_title("Current separation on the LJ curve")
        potential_axis.grid(alpha=0.2)
        potential_axis.legend(loc="lower right", frameon=False)

        energy_change = total_energy - total_energy[0]
        drift_axis.plot(times, energy_change, color="#6a4c93", lw=2.2)
        drift_axis.axhline(0, color="0.25", ls="--", lw=1.2)
        drift_axis.scatter([current_time], [energy_change[sample_index]],
                           color="#e36c2f", edgecolor="white", s=75, zorder=3)
        drift_axis.set_xlim(0, max(times[-1], dt))
        drift_axis.set_xlabel("Time t (τ)")
        drift_axis.set_ylabel("E(t) − E(0) (ε)")
        drift_axis.tick_params(labelsize=16)
        drift_axis.set_title(f"Energy drift · {integrator.value} · dt = {dt:g}")
        drift_axis.grid(alpha=0.22)

    current_momentum = float(np.max(np.abs(np.sum(velocities[sample_index], axis=0))))
    initial_residual_text = (
        f"turning-root residual {root_residual:.1e} ε"
        if root_residual is not None else "kick at the potential minimum"
    )
    status = mo.md(
        f"**Start:** {preparation.value}, r = {start_separation:.4f} σ, "
        f"E₀ = {target_energy:.4f} ε; {initial_residual_text}; "
        f"initial energy residual {start_energy_residual:.1e} ε.  \n\n"
        f"**t = {current_time:.3f} τ** · **r = {separation:.4f} σ** · "
        f"**U = {current_potential:.5f} ε** · **K = {current_kinetic:.5f} ε** · "
        f"**E = {current_total:.5f} ε** · **|P| = {current_momentum:.2g}**  \n\n"
        f"**Maximum |E − E(0)|:** {max_energy_drift:.3g} ε · "
        f"**maximum |P|:** {max_momentum:.2g}."
    )
    warning = mo.md(
        "**The trajectory stopped at the safety limit.** Reduce dt and rerun."
        if stopped else ""
    )
    mo.vstack([status, figure, warning])
    return (figure,)


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.12", "plotly>=7.1.0"]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import optimize

    return go, make_subplots, mo, np, optimize


@app.cell(hide_code=True)
def _(np):
    # Argon LJ parameters convert the reduced time unit τ to femtoseconds.
    tau_fs = 3.405e-10 * np.sqrt(39.948 * 1.66054e-27 / (119.8 * 1.380649e-23)) * 1e15
    return (tau_fs,)


@app.cell(hide_code=True)
def _(dt, duration, mo, n_steps, tau_fs):
    mo.md(
        "## Refined LJ Cluster Simulation with Velocity Verlet\n\n"
        f"Total time steps: **{n_steps:,}** · "
        f"single step Δt = **{dt * tau_fs:.0f} fs** = {dt:.5f} τ · "
        f"total time {duration * tau_fs / 1000:.0f} ps = {duration:.2f} τ "
        f"(argon: σ = 3.405 Å, ε/k<sub>B</sub> = 119.8 K, 1 τ = {tau_fs:.0f} fs)"
    )
    return


@app.cell
def _(np):
    def energy_func(r):
        # Lennard–Jones pair potential in reduced units.
        return 4 * (r**-12 - r**-6)

    def U(pos):
        # Sum every unique pair i < j, as in the L21 MD engine.
        atom_i, atom_j = np.triu_indices(len(pos), k=1)
        separation = np.linalg.norm(pos[atom_i] - pos[atom_j], axis=1)
        return np.sum(energy_func(separation))

    def forces(pos):
        # Analytical negative gradient of U, with equal and opposite pair forces.
        atom_i, atom_j = np.triu_indices(len(pos), k=1)
        displacement = pos[atom_i] - pos[atom_j]
        separation = np.linalg.norm(displacement, axis=1)
        pair_force = (
            24 * (2 * separation**-14 - separation**-8)
        )[:, None] * displacement
        force = np.zeros_like(pos)
        np.add.at(force, atom_i, pair_force)
        np.add.at(force, atom_j, -pair_force)
        return force

    return U, energy_func, forces


@app.cell
def _(U, forces, np, optimize, params):
    # Relax the first N sites of a seeded 13-site icosahedral guess; centre at zero.
    phi = (1 + np.sqrt(5)) / 2
    shell = []
    for a in [-1, 1]:
        for b in [-phi, phi]:
            shell.extend([[0, a, b], [a, b, 0], [b, 0, a]])
    pos_guess = 0.6 * np.vstack([np.zeros((1, 3)), np.array(shell)])
    pos_guess += 0.05 * np.random.default_rng(21).normal(size=pos_guess.shape)
    pos_guess = pos_guess[:params["atoms"]]
    pos_guess -= pos_guess.mean(axis=0)

    result = optimize.minimize(
        lambda x: U(x.reshape(-1, 3)),
        pos_guess.ravel(),
        jac=lambda x: -forces(x.reshape(-1, 3)).ravel(),
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 600},
    )
    pos_min = result.x.reshape(-1, 3)
    pos_min -= pos_min.mean(axis=0)
    U_min = U(pos_min)
    return U_min, pos_min


@app.cell(hide_code=True)
def _(mo):
    settings = mo.md(
        "{atoms} &nbsp; {added} &nbsp; {preparation}\n\n{dt_fs} &nbsp; {total_ps}"
    ).batch(
        atoms=mo.ui.slider(
            start=2, stop=13, step=1, value=13,
            label="Number of atoms", show_value=True,
        ),
        added=mo.ui.slider(
            start=0.10, stop=6.00, step=0.10, value=0.30,
            label="Added energy per atom (ε)", show_value=True,
        ),
        preparation=mo.ui.dropdown(
            ["Random velocities at relaxed structure",
             "Compress a nearest-neighbour pair, release at rest"],
            value="Random velocities at relaxed structure",
            label="Preparation",
        ),
        dt_fs=mo.ui.dropdown(
            {"1 fs": 1.0, "2 fs": 2.0, "5 fs": 5.0, "10 fs": 10.0, "20 fs": 20.0},
            value="2 fs", label="Time step Δt",
        ),
        total_ps=mo.ui.slider(
            start=2, stop=20, step=1, value=16,
            label="Total simulated time (ps)", show_value=True,
        ),
    ).form(submit_button_label="Run simulation", bordered=False)
    settings
    return (settings,)


@app.cell
def _(settings):
    # The simulation runs only when the form is submitted; before that, use the defaults.
    params = settings.value or {
        "atoms": 13,
        "added": 0.30,
        "preparation": "Random velocities at relaxed structure",
        "dt_fs": 2.0,
        "total_ps": 16,
    }
    return (params,)


@app.cell
def _(U, U_min, np, optimize, params, pos_min):
    def initial_state(added_energy, preparation):
        atom_count = len(pos_min)
        added_total = atom_count * added_energy
        target_energy = U_min + added_total

        if preparation == "Random velocities at relaxed structure":
            start_pos = pos_min.copy()
            start_vel = np.random.default_rng(374).normal(size=pos_min.shape)
            start_vel -= start_vel.mean(axis=0)
            kinetic_target = target_energy - U(start_pos)
            start_vel *= np.sqrt(kinetic_target / (0.5 * np.sum(start_vel**2)))
            root_residual = 0.0
        else:
            # Move the closest pair symmetrically along its bond until U reaches target.
            atom_i, atom_j = np.triu_indices(atom_count, k=1)
            separations = np.linalg.norm(pos_min[atom_i] - pos_min[atom_j], axis=1)
            pair_index = int(np.argmin(separations))
            i, j = atom_i[pair_index], atom_j[pair_index]
            bond = pos_min[i] - pos_min[j]
            bond_length = separations[pair_index]
            bond_direction = bond / bond_length

            def compressed_state(fraction):
                compressed = pos_min.copy()
                shift = 0.5 * fraction * bond_length * bond_direction
                compressed[i] -= shift
                compressed[j] += shift
                return compressed

            residual = lambda fraction: U(compressed_state(fraction)) - target_energy
            upper = 0.99
            scale = optimize.brentq(residual, 0.0, upper, xtol=1e-12)
            start_pos = compressed_state(scale)
            start_vel = np.zeros_like(pos_min)
            root_residual = abs(residual(scale))

        energy_residual = abs(U(start_pos) + 0.5 * np.sum(start_vel**2) - target_energy)
        return start_pos, start_vel, target_energy, root_residual, energy_residual

    pos0, vel0, target_energy, root_residual, initial_energy_residual = initial_state(
        params["added"], params["preparation"]
    )
    return pos0, vel0


@app.cell
def _(forces, np):
    def velocity_verlet_step(pos, vel, force, dt, mass=1.0):
        # Position-first velocity Verlet; return the new force for the next step.
        vel_half = vel + 0.5 * dt * force / mass
        pos_next = pos + dt * vel_half
        force_next = forces(pos_next)
        vel_next = vel_half + 0.5 * dt * force_next / mass
        return pos_next, vel_next, force_next

    def run(pos0, vel0, dt, n_steps, mass=1.0):
        pos = pos0.copy()
        vel = vel0.copy()
        force = forces(pos)
        positions = np.empty((n_steps + 1, *pos.shape))
        velocities = np.empty_like(positions)
        positions[0], velocities[0] = pos, vel
        completed = 0

        for step_index in range(n_steps):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                pos_next, vel_next, force_next = velocity_verlet_step(
                    pos, vel, force, dt, mass
                )
            if (not np.all(np.isfinite(pos_next))
                    or not np.all(np.isfinite(vel_next))
                    or np.max(np.abs(pos_next)) > 1e4):
                break
            positions[step_index + 1] = pos_next
            velocities[step_index + 1] = vel_next
            pos, vel, force = pos_next, vel_next, force_next
            completed += 1

        stopped = completed < n_steps
        return positions[:completed + 1], velocities[:completed + 1], stopped

    return (run,)


@app.cell
def _(U, np, params, pos0, run, tau_fs, vel0):
    # Convert the physical inputs (fs, ps) to reduced LJ time units.
    dt = params["dt_fs"] / tau_fs
    n_steps = int(round(1000 * params["total_ps"] / params["dt_fs"]))
    duration = n_steps * dt
    positions, velocities, stopped = run(pos0, vel0, dt, n_steps)

    times = np.arange(len(positions)) * dt
    potential_energy = np.array([U(pos) for pos in positions])
    kinetic_energy = 0.5 * np.sum(velocities**2, axis=(1, 2))
    total_energy = potential_energy + kinetic_energy
    momentum = np.sum(velocities, axis=1)
    max_momentum = np.max(np.linalg.norm(momentum, axis=1))
    max_energy_error = np.max(np.abs(total_energy - total_energy[0]))
    display_indices = np.unique(np.linspace(
        0, len(times) - 1, min(150, len(times)), dtype=int
    ))
    simulation_warning = (
        "**Run stopped at a numerical safety limit.**"
        if stopped else ""
    )
    return (
        display_indices,
        dt,
        duration,
        kinetic_energy,
        n_steps,
        positions,
        potential_energy,
        simulation_warning,
        times,
        total_energy,
    )


@app.cell
def _(energy_func, go, make_subplots, np):
    atom_colors = [
        "#1f77b4",
    ]

    def nearest_neighbours(pos):
        displacement = pos[:, None, :] - pos[None, :, :]
        separation = np.linalg.norm(displacement, axis=2)
        np.fill_diagonal(separation, np.inf)
        partner = np.argmin(separation, axis=1)
        distance = separation[np.arange(len(pos)), partner]
        return distance, partner

    def current_title(time, potential, kinetic, total):
        return (
            f"<b>t = {time:.2f} ps &nbsp; | &nbsp; "
            f"U<sub>cluster</sub> = {potential:.4f} &nbsp; | &nbsp; "
            f"K = {kinetic:.4f} &nbsp; | &nbsp; E = {total:.4f}</b>"
        )

    def rgba(hex_color, alpha):
        red = int(hex_color[1:3], 16)
        green = int(hex_color[3:5], 16)
        blue = int(hex_color[5:7], 16)
        return f"rgba({red},{green},{blue},{alpha})"

    def depth_size(z, view_limit):
        # Larger markers are closer to the viewer (positive z), scaled to the view box.
        return np.clip(20 + 8 * z / view_limit, 12, 28)

    def make_figure(positions, times, potential_energy, kinetic_energy,
                    total_energy, display_indices):
        distances_all = np.array([nearest_neighbours(pos)[0] for pos in positions])
        maximum_distance = float(np.max(distances_all))
        distance_max = max(2.5, 1.1 * maximum_distance)
        use_log = distance_max > 4.0
        distance_curve = (np.geomspace(0.85, distance_max, 500) if use_log
                          else np.linspace(0.85, distance_max, 500))
        pair_curve = energy_func(distance_curve)
        view_limit = max(3.3, 1.12 * float(np.max(np.abs(positions))))
        atom_ids = np.arange(1, positions.shape[1] + 1)
        marker_colors = rgba(atom_colors[0], 0.35)
        initial_distance, initial_partner = nearest_neighbours(positions[0])
        initial_pair_energy = energy_func(initial_distance)
        initial_customdata = np.column_stack([
            atom_ids, initial_partner + 1, initial_distance
        ])

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}]],
            column_widths=[0.42, 0.58],
            horizontal_spacing=0.08,
            subplot_titles=("Cluster (x–y view, larger = nearer)", "Nearest-neighbour pair energy"),
        )
        fig.add_trace(go.Scatter(
            x=positions[0, :, 0], y=positions[0, :, 1],
            mode="markers",
            customdata=atom_ids[:, None],
            marker={"size": depth_size(positions[0, :, 2], view_limit), "color": "#8a8f98",
                    "gradient": {"type": "radial", "color": "white"},
                    "line": {"color": "#4b5058", "width": 1.0}},
            hovertemplate="atom %{customdata[0]}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=distance_curve, y=pair_curve, mode="lines",
            line={"color": "#111827", "width": 3},
            name="Lennard–Jones pair potential",
            hovertemplate="r=%{x:.4f}<br>V(r)=%{y:.4f}<extra></extra>",
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=initial_distance, y=initial_pair_energy, mode="markers",
            customdata=initial_customdata,
            marker={"size": 11, "color": marker_colors,
                    "line": {"width": 0.4, "color": "#17212b"}},
            name="atoms",
            hovertemplate=(
                "atom %{customdata[0]} → nearest atom %{customdata[1]}"
                "<br>rᵢ=%{x:.4f}<br>pair V(rᵢ)=%{y:.4f}<extra></extra>"
            ),
            showlegend=False,
        ), row=1, col=2)

        frames = []
        for frame_number, index in enumerate(display_indices):
            neighbour_distance, neighbour_partner = nearest_neighbours(positions[index])
            neighbour_energy = energy_func(neighbour_distance)
            customdata = np.column_stack([
                atom_ids, neighbour_partner + 1, neighbour_distance
            ])
            frame_data = [
                go.Scatter(
                    x=positions[index, :, 0],
                    y=positions[index, :, 1],
                    customdata=atom_ids[:, None],
                    marker={"size": depth_size(positions[index, :, 2], view_limit)},
                ),
                go.Scatter(
                    x=neighbour_distance,
                    y=neighbour_energy,
                    customdata=customdata,
                ),
            ]
            frames.append(go.Frame(
                name=str(frame_number),
                data=frame_data,
                traces=[0, 2],
                layout=go.Layout(title_text=current_title(
                    times[index], potential_energy[index],
                    kinetic_energy[index], total_energy[index]
                )),
            ))

        fig.frames = frames

        slider_steps = []
        for frame_number, index in enumerate(display_indices):
            label = f"{times[index]:.1f}" if frame_number % 15 == 0 else ""
            slider_steps.append({
                "method": "animate",
                "label": label,
                "args": [[str(frame_number)], {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                }],
            })

        initial_index = display_indices[0]
        fig.update_layout(
            title={
                "text": current_title(
                    times[initial_index], potential_energy[initial_index],
                    kinetic_energy[initial_index], total_energy[initial_index]
                ),
                "x": 0.0, "xanchor": "left", "y": 0.98,
                "font": {"size": 16, "color": "#17212b"},
            },
            autosize=True,
            height=470,
            modebar={"orientation": "v"},
            margin={"l": 20, "r": 30, "t": 110, "b": 90},
            paper_bgcolor="white",
            plot_bgcolor="#fbfcfd",
            hoverlabel={"font": {"size": 13}},
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "x": 0.02, "y": -0.14,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": 65, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        }],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [[None], {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": False},
                            "transition": {"duration": 0},
                        }],
                    },
                ],
            }],
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "time (ps): ", "font": {"size": 12}},
                "pad": {"t": 42, "b": 6},
                "x": 0.26, "y": -0.14, "len": 0.70,
                "steps": slider_steps,
            }],
            xaxis={"title": "x (σ)", "range": [-view_limit, view_limit],
                   "gridcolor": "#e2e8ec", "zeroline": False},
            yaxis={"title": "y (σ)", "range": [-view_limit, view_limit],
                   "gridcolor": "#e2e8ec", "zeroline": False,
                   "scaleanchor": "x", "scaleratio": 1},
            xaxis2={
                "title": "nearest-neighbour distance r (σ)",
                "type": "log" if use_log else "linear",
                "range": ([np.log10(0.85), np.log10(distance_max)] if use_log
                          else [0.85, distance_max]),
                "gridcolor": "#e2e8ec",
                "zeroline": False,
            },
            yaxis2={
                "title": "pair energy V(r) (ε)",
                "range": [-1.1, 0.4],
                "gridcolor": "#e2e8ec",
                "zeroline": False,
            },
            showlegend=False,
        )
        return fig

    return (make_figure,)


@app.cell
def _(
    display_indices,
    kinetic_energy,
    make_figure,
    mo,
    positions,
    potential_energy,
    simulation_warning,
    tau_fs,
    times,
    total_energy,
):
    animation = make_figure(
        positions, times * tau_fs / 1000, potential_energy, kinetic_energy,
        total_energy, display_indices
    )
    mo.vstack([animation, mo.md(simulation_warning)] if simulation_warning
              else [animation])
    return


if __name__ == "__main__":
    app.run()

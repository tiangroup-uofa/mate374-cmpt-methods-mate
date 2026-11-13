# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ase==3.29.0",
#     "marimo==0.25.2",
#     "matplotlib==3.11.1",
#     "numpy==2.4.6",
#     "scipy==1.18.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analyse your Cu nanoparticle simulations

    1. Upload **one CSV** to plot the energies and identify a melting temperature.
    2. Upload **one XYZ** to inspect the particle at different temperatures.
    3. Compare the pair distribution and the solid-like fraction at the selected frame.
    4. Record your measurements in Table 3 and complete the size-dependent fit.

    The CSV and XYZ upload boxes are separate. Replace a file to analyse another run.
    """)
    return


@app.cell(hide_code=True)
def _():
    import io
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.io import read
    from ase.visualize.plot import plot_atoms
    from matplotlib.patches import Patch, Circle
    from scipy.spatial import cKDTree, ConvexHull, QhullError
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.special import sph_harm_y
    from scipy.optimize import minimize

    KB_EV = 8.617333262e-5  # eV/K
    return (
        Circle,
        ConvexHull,
        KB_EV,
        Patch,
        QhullError,
        cKDTree,
        connected_components,
        coo_matrix,
        io,
        mo,
        np,
        plot_atoms,
        plt,
        read,
        sph_harm_y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Energy from one CSV file

    Upload the energy file from either experiment in notebook 1.

    - **Fixed temperature:** compare the potential and kinetic energies against time.
    - **Heating and cooling:** compare the potential energy against the thermostat temperature.

    Locate the change in the heating curve that you will use to identify melting.
    Record its temperature and your criterion in your report. Use the XYZ snapshots below to check the structural change.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    csv_upload = mo.ui.file(filetypes=[".csv"], multiple=False, kind="area",
                            label="Upload one energy CSV")
    csv_upload
    return (csv_upload,)


@app.cell(hide_code=True)
def _(csv_upload, load_run, mo, np):
    mo.stop(not csv_upload.value, mo.md("Upload one CSV above to draw its energy curves."))
    csv_error = None
    try:
        csv_data = load_run(csv_upload.value[0].contents.decode("utf-8-sig"))
        required_columns = {"time_ps", "stage", "T_set_K", "PE_eV_per_atom", "KE_eV_per_atom"}
        if not required_columns.issubset(csv_data.dtype.names or ()):
            csv_error = "Use a simulation CSV with time_ps, stage, T_set_K, PE_eV_per_atom and KE_eV_per_atom columns."
        elif len(csv_data) == 0:
            csv_error = "The CSV has no data rows."
        elif not all(np.all(np.isfinite(csv_data[name])) for name in required_columns - {"stage"}):
            csv_error = "The CSV contains missing or non-finite numerical values."
    except (ValueError, TypeError, UnicodeError) as error:
        csv_error = f"Could not read this CSV: {error}"
    mo.stop(csv_error is not None, mo.md(csv_error or ""))
    mo.md(f"**{csv_upload.value[0].name}** · {len(csv_data)} records")
    return (csv_data,)


@app.cell(hide_code=True)
def _(KB_EV, csv_data, mo, np, plt):
    has_cycle = np.any(csv_data["stage"] == "heat")
    if has_cycle:
        energy_fig, energy_axis = plt.subplots(figsize=(7, 4))
        for stage_name, stage_colour in (("heat", "tab:red"), ("cool", "tab:blue")):
            stage_mask = csv_data["stage"] == stage_name
            if stage_mask.any():
                energy_axis.plot(csv_data["T_set_K"][stage_mask],
                                 csv_data["PE_eV_per_atom"][stage_mask],
                                 color=stage_colour, lw=1, label="Heating" if stage_name == "heat" else "Cooling")
        energy_axis.set(xlabel="Thermostat temperature (K)", ylabel="Potential energy (eV/atom)")
        energy_axis.legend(frameon=False)
        energy_caption = mo.md("**Heating and cooling:** follow the potential-energy change as the thermostat temperature changes. "
                               "Identify the melting feature on heating and compare the atomic snapshots at nearby temperatures.")
    else:
        energy_fig, energy_axes = plt.subplots(1, 2, figsize=(10, 3.5))
        energy_axes[0].plot(csv_data["time_ps"], csv_data["PE_eV_per_atom"], color="tab:red", lw=1)
        energy_axes[0].set(xlabel="Time (ps)", ylabel="Potential energy (eV/atom)")
        energy_axes[1].plot(csv_data["time_ps"], csv_data["KE_eV_per_atom"],
                            color="tab:blue", lw=1, label="Kinetic energy")
        energy_axes[1].plot(csv_data["time_ps"], 1.5 * KB_EV * csv_data["T_set_K"],
                            color="black", ls="--", lw=1, label="Expected mean at target T")
        energy_axes[1].set(xlabel="Time (ps)", ylabel="Kinetic energy (eV/atom)")
        energy_axes[1].legend(frameon=False, fontsize=8)
        energy_caption = mo.md(r"**Fixed temperature:** the dashed line is the expected equilibrium mean kinetic energy per atom, "
                               r"\(\tfrac32 k_\mathrm{B}T\), at the thermostat temperature \(T\). "
                               r"Here \(k_\mathrm{B}=8.6173\times10^{-5}\) eV/K is Boltzmann's constant. "
                               "The instantaneous energy fluctuates around this mean. Compare it after the initial transient.")
    energy_fig.tight_layout()
    mo.vstack([energy_fig, energy_caption])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    For the fixed-temperature run, identify when both energies fluctuate about stable means. "
          "For a cycle, note the melting and freezing features separately. "
          "The uploaded CSV supplies the plotted temperatures; it does not assign a melting point automatically.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Atomic snapshots from one XYZ file

    Upload the structure file from the heating-and-cooling run. The slider selects a saved frame.

    - **Whole particle:** ASE draws the atoms without an external 3D viewer.
    - **Central slice:** atoms within 2 Å of the centre plane reveal the interior.
    - Each picture shows the frame's stage, thermostat temperature, and time.

    The six-picture summary below is ready to use as a starting point for a report figure.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    xyz_upload = mo.ui.file(filetypes=[".xyz"], multiple=False, kind="area",
                            label="Upload one atomic-structure XYZ")
    xyz_upload
    return (xyz_upload,)


@app.cell(hide_code=True)
def _(io, mo, np, read, xyz_upload):
    mo.stop(not xyz_upload.value, mo.md("Upload one XYZ above to inspect its snapshots."))
    xyz_error = None
    try:
        frames = read(io.StringIO(xyz_upload.value[0].contents.decode("utf-8-sig")),
                      index=":", format="extxyz")
        if not frames or any(len(atoms) == 0 for atoms in frames):
            xyz_error = "The XYZ must contain at least one nonempty atomic structure."
        elif any(not np.all(np.isfinite(atoms.positions)) for atoms in frames):
            xyz_error = "The XYZ contains non-finite atom positions."
    except (ValueError, TypeError, UnicodeError, IndexError, KeyError) as error:
        xyz_error = f"Could not read this XYZ: {error}"
    mo.stop(xyz_error is not None, mo.md(xyz_error or ""))
    mo.md(f"**{xyz_upload.value[0].name}** · {len(frames)} saved frames")
    return (frames,)


@app.cell(hide_code=True)
def _(frames, mo):
    frame_slider = mo.ui.slider(0, max(1, len(frames) - 1), value=0, step=1,
                                disabled=len(frames) == 1, show_value=True,
                                label="Saved frame")
    frame_slider
    return (frame_slider,)


@app.cell(hide_code=True)
def _(
    draw_particle,
    frame_label,
    frame_slider,
    frames,
    largest_particle,
    mo,
    plt,
):
    selected_frame = frames[frame_slider.value]
    selected_centred, selected_bound = largest_particle(selected_frame.positions)
    structure_fig, structure_axes = plt.subplots(1, 2, figsize=(8, 3.8))
    draw_particle(selected_frame, structure_axes[0])
    draw_particle(selected_frame, structure_axes[1], central_slice=True)
    structure_axes[0].set_title("Whole particle")
    structure_axes[1].set_title("Central slice")
    structure_fig.suptitle(frame_label(selected_frame), fontsize=11)
    structure_fig.tight_layout()
    mo.vstack([
        mo.md(f"**Main particle:** {int(selected_bound.sum())} atoms · "
              f"**Detached:** {int((~selected_bound).sum())}. "
              "The slice and structural estimates use the main connected particle (3.8 Å connection cutoff)."),
        structure_fig,
    ])
    return selected_bound, selected_centred, selected_frame


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Selected temperatures in one figure

    "
          "The summary chooses up to five saved heating frames spread across the temperature range, "
          "followed by the final frame. Labels retain the actual saved temperatures and times.
    """)
    return


@app.cell(hide_code=True)
def _(draw_particle, frame_label, frames, np, plt):
    heating_indices = [index for index, atoms in enumerate(frames)
                       if atoms.info.get("stage") == "heat"]
    snapshot_indices = []
    if heating_indices:
        heating_temperatures = np.array([float(frames[index].info.get("T_set_K", np.nan))
                                         for index in heating_indices])
        if np.all(np.isfinite(heating_temperatures)):
            targets = np.linspace(heating_temperatures.min(), heating_temperatures.max(), 5)
            snapshot_indices = list(dict.fromkeys(
                heating_indices[int(np.argmin(abs(heating_temperatures - target)))] for target in targets
            ))
    if not snapshot_indices:
        snapshot_indices = list(dict.fromkeys(np.linspace(0, len(frames) - 1, min(5, len(frames))).astype(int)))
    snapshot_indices = list(dict.fromkeys(snapshot_indices + [len(frames) - 1]))
    snapshot_fig, snapshot_axes = plt.subplots(2, 3, figsize=(10, 6))
    for snapshot_axis in snapshot_axes.flat:
        snapshot_axis.set_axis_off()
    for snapshot_index, snapshot_axis in zip(snapshot_indices, snapshot_axes.flat):
        draw_particle(frames[snapshot_index], snapshot_axis, central_slice=True)
        snapshot_axis.set_title(frame_label(frames[snapshot_index]), fontsize=8)
    snapshot_fig.suptitle("Central slices at saved temperatures", fontsize=12)
    snapshot_fig.tight_layout()
    snapshot_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Pair distribution and solid-like fraction

    The **pair distribution function** \(g(r)\) compares the number of neighbours at distance \(r\)
    with the number expected at the same average density without structural order.

    - **Crystal:** sharp peaks appear at regular neighbour distances.
    - **Liquid:** the first peak is broader and the later peaks are weaker.
    - Move the frame slider above to compare the selected structure with the initial one.

    The coloured central slice also uses the directions of neighbouring bonds to label each atom.
    Gold means **solid-like** and blue means **liquid-like** by this local-order criterion.
    The fractions are estimates from those labels, rather than a direct measurement from \(g(r)\).
    Atoms at a free surface can also receive a liquid-like label.

    The neighbour-distance plot uses interior atoms with complete spherical shells inside the
    main particle's convex envelope. Its range is shortened for a small particle.
    The density normalization uses that envelope's volume, so the absolute peak heights are approximate.
    """)
    return


@app.cell(hide_code=True)
def _(
    Circle,
    Patch,
    draw_particle,
    frame_label,
    frame_slider,
    frames,
    mo,
    np,
    pair_distribution,
    plt,
    radial_geometry,
    selected_bound,
    selected_centred,
    selected_frame,
    solid_like,
):
    selected_solid = solid_like(selected_centred[selected_bound])
    selected_colours = np.full(len(selected_frame), "tab:blue", dtype=object)
    selected_colours[selected_bound] = np.where(selected_solid, "goldenrod", "tab:blue")
    order_fig, order_axes = plt.subplots(1, 2, figsize=(10, 3.8))
    slice_origin = draw_particle(selected_frame, order_axes[1], central_slice=True, colours=selected_colours)
    order_axes[1].plot(*slice_origin, marker="+", color="black", ms=12, mew=2, zorder=8)
    order_axes[1].annotate("Particle centre", xy=slice_origin,
                           xytext=(0.98, 0.05), textcoords="axes fraction", ha="right", fontsize=8,
                           arrowprops={"arrowstyle": "-", "color": "black"}, zorder=9)
    order_axes[1].set_title("Central slice")
    order_axes[1].legend(handles=[Patch(color="goldenrod", label="solid-like"),
                                  Patch(color="tab:blue", label="liquid-like")],
                         frameon=False, fontsize=8)
    radial_message = ""
    try:
        initial_geometry = radial_geometry(frames[0].positions)
        current_geometry = radial_geometry(selected_frame.positions)
        radial_cutoff = min(8.0, 0.7 * initial_geometry[1].max(), 0.7 * current_geometry[1].max())
        if radial_cutoff < 2.8:
            raise ValueError("This structure is too small for a useful neighbour-distance range.")
        for atoms_for_gr, curve_label, curve_colour in (
                (frames[0], "Initial", "grey"), (selected_frame, "Selected", "tab:red")):
            distances_gr, values_gr = pair_distribution(atoms_for_gr.positions, radial_cutoff)
            order_axes[0].plot(distances_gr, values_gr, label=curve_label, color=curve_colour, lw=1)
        order_axes[0].set(xlabel="Distance r (Å)", ylabel="g(r)", xlim=(2, radial_cutoff))
        order_axes[0].legend(frameon=False, fontsize=8)
        reference_candidates = np.flatnonzero(current_geometry[1] > radial_cutoff)
        reference_atom = reference_candidates[np.argmin(np.linalg.norm(current_geometry[0][reference_candidates], axis=1))]
        reference_xy = current_geometry[0][reference_atom, :2] + slice_origin
        order_axes[1].plot(*reference_xy, marker="o", color="purple", ms=5, zorder=10)
        order_axes[1].add_patch(Circle(reference_xy, radial_cutoff, fill=False,
                                      edgecolor="purple", linestyle="--", linewidth=1.2, zorder=7))
        arrow_tip = reference_xy + radial_cutoff * np.array([np.cos(np.pi/6), np.sin(np.pi/6)])
        order_axes[1].annotate("", xy=arrow_tip, xytext=reference_xy,
                               arrowprops={"arrowstyle": "->", "color": "purple", "lw": 1.5}, zorder=9)
        order_axes[1].text(0.98, 0.97, f"r max = {radial_cutoff:.2f} Å",
                           transform=order_axes[1].transAxes, ha="right", va="top", fontsize=8, color="purple")
        radial_message = (f"Neighbour-distance range: {radial_cutoff:.2f} Å. "
                          "The black cross marks the particle centre. The purple dot is an example interior reference atom. "
                          "Its dashed circle and arrow show the neighbour-distance cutoff, projected onto the slice.")
    except ValueError as error:
        order_axes[0].text(0.5, 0.5, str(error), ha="center", va="center", wrap=True,
                           transform=order_axes[0].transAxes, fontsize=9)
        order_axes[0].set_axis_off()
    order_fig.suptitle(frame_label(selected_frame), fontsize=11)
    order_fig.tight_layout()
    mo.vstack([
        frame_slider,
        mo.md(f"**Main particle:** {selected_solid.mean():.1%} solid-like · "
              f"{1 - selected_solid.mean():.1%} liquid-like by the local-order criterion. {radial_message}"),
        order_fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Particle diameter and melting temperature

    ### Table 3 · Template for your melting-temperature measurements

    Enter the heating rate and the melting temperature measured for each spherical particle.
    The diameters are example settings. The melting-temperature cells are deliberately blank for your measurements. Edit the table's code if you need more rows.
    Use the same heating rate for the size-dependent fit, after comparing rates in section 1.

    The fitting law is

    \[
    T_\mathrm{m}(D)=T_\mathrm{m,bulk}\left[1-\left(\frac{D_0}{D}\right)^\alpha\right].
    \]

    - \(D\): particle diameter in nm.
    - \(D_0\): fitted length in nm.
    - \(\alpha\): fitted dimensionless exponent.
    - \(T_\mathrm{m,bulk}\): fitted bulk melting temperature in K.

    Complete **fit_melting_data** below with **scipy.optimize.minimize**.
    The parameter order is **bulk temperature, \(D_0\), \(\alpha\)**.
    Choose initial guesses and positive bounds, with \(D_0\) below the smallest fitted diameter.
    Minimize the sum of squared temperature residuals.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    measurements = mo.ui.data_editor(
        [{"Diameter_nm": diameter, "Rate_K_ps": "", "Melting_K": ""}
         for diameter in (2.0, 3.0, 4.0, 6.0, 8.0)],
        label="Table 3 — spherical particles; enter your own rates and melting temperatures",
        pagination=False,
    )
    measurements
    return (measurements,)


@app.cell
def melting_fit_exercise(np):
    def melting_law(diameter_nm, parameters):
        bulk_temperature, length_nm, exponent = parameters
        return bulk_temperature * (1 - (length_nm / np.asarray(diameter_nm))**exponent)

    def melting_objective(parameters, diameter_nm, melting_K):
        residual = melting_law(diameter_nm, parameters) - melting_K
        return np.sum(residual**2)

    def fit_melting_data(diameter_nm, melting_K):
        # Choose initial guesses for [bulk temperature (K), D0 (nm), alpha].
        # Choose bounds that keep all parameters positive and D0 below min(diameter_nm).
        # Replace the next line with minimize(melting_objective, ..., args=(diameter_nm, melting_K), ...).
        return None


    return fit_melting_data, melting_law


@app.cell(hide_code=True)
def melting_fit_result(
    fit_melting_data,
    measurements,
    melting_law,
    mo,
    np,
    plt,
):
    fit_rows = []
    invalid_rows = []
    for row_index, row_value in enumerate(measurements.value, start=1):
        if str(row_value["Melting_K"]).strip() == "":
            continue
        try:
            numeric_row = tuple(float(row_value[key]) for key in ("Diameter_nm", "Rate_K_ps", "Melting_K"))
            if not all(np.isfinite(value) and value > 0 for value in numeric_row):
                raise ValueError("positive finite values required")
            fit_rows.append(numeric_row)
        except (ValueError, TypeError):
            invalid_rows.append(row_index)
    mo.stop(bool(invalid_rows), mo.md(f"Enter positive numerical diameter, rate and temperature values in rows {invalid_rows}."))
    mo.stop(not fit_rows, mo.md("Enter your measured melting temperatures and heating rates in Table 3."))
    fit_array = np.asarray(fit_rows)
    mo.stop(len(np.unique(fit_array[:, 1])) != 1,
            mo.md("Use one heating rate for this fit. Compare rates separately before choosing the size-study rate."))
    diameter_data = fit_array[:, 0]
    melting_data = fit_array[:, 2]
    fit_result = None
    if len(np.unique(diameter_data)) >= 4:
        fit_result = fit_melting_data(diameter_data, melting_data)
    fit_fig, fit_axes = plt.subplots(figsize=(6, 3.6))
    fit_axes.plot(diameter_data, melting_data, "o", label="Your measurements")
    fit_message = "Complete fit_melting_data to add the fitted curve and parameters."
    if len(np.unique(diameter_data)) < 4:
        fit_message = "Use at least four distinct diameters for a three-parameter fit with a residual check."
    elif fit_result is not None:
        if not fit_result.success:
            fit_message = f"The optimizer did not converge: {fit_result.message}"
        elif np.shape(fit_result.x) != (3,) or not np.all(np.isfinite(fit_result.x)):
            fit_message = "The optimizer must return three finite parameters."
        elif not (fit_result.x[0] > 0 and 0 < fit_result.x[1] < diameter_data.min() and fit_result.x[2] > 0):
            fit_message = "Check the physical bounds: positive bulk temperature and exponent, and 0 < D0 < the smallest diameter."
        else:
            fitted_diameters = np.linspace(diameter_data.min(), diameter_data.max(), 200)
            fit_axes.plot(fitted_diameters, melting_law(fitted_diameters, fit_result.x), label="Your fitted law")
            fit_message = (f"**Fitted bulk melting temperature:** {fit_result.x[0]:.2f} K · "
                           f"**D0:** {fit_result.x[1]:.4f} nm · **alpha:** {fit_result.x[2]:.4f}. "
                           "Compare the bulk estimate with 1357.77 K and inspect the fit to your measurements.")
    fit_axes.set(xlabel="Particle diameter (nm)", ylabel="Melting temperature (K)")
    fit_axes.legend(frameon=False)
    fit_fig.tight_layout()
    mo.vstack([mo.md(fit_message), fit_fig])
    return


@app.cell(hide_code=True)
def report_figures(mo):
    mo.md("""
    ### Report figures

    "
          "The plots are editable Matplotlib figures. Save a report figure at 300 dpi, "
          "for example `snapshot_fig.savefig("snapshots.png", dpi=300, bbox_inches="tight")`. "
          "State how you identified each melting temperature and which heating rate you used.
    """)
    return


@app.cell(hide_code=True)
def _(io, np):
    def load_run(text):
        """Read the named columns in a simulation CSV, including its text stage column."""
        return np.atleast_1d(np.genfromtxt(
            io.StringIO(text), delimiter=",", names=True, dtype=None, encoding="utf-8"
        ))

    return (load_run,)


@app.cell(hide_code=True)
def _(cKDTree, connected_components, coo_matrix, np, plot_atoms):
    def largest_particle(positions, connection_A=3.8):
        """Centre on the largest group of atoms connected below connection_A."""
        pairs = cKDTree(positions).query_pairs(connection_A, output_type="ndarray")
        graph = coo_matrix((np.ones(2 * len(pairs)),
                           (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                          shape=(len(positions), len(positions)))
        _, labels = connected_components(graph, directed=False)
        bound = labels == np.argmax(np.bincount(labels))
        return positions - positions[bound].mean(axis=0), bound

    def frame_label(atoms):
        """Read saved simulation metadata, without inferring missing temperatures."""
        return (f"{atoms.info.get('stage', 'structure')}, "
                f"{atoms.info.get('T_set_K', '?')} K, "
                f"{atoms.info.get('time_ps', '?')} ps")

    def draw_particle(atoms, axes, central_slice=False, colours=None):
        centred, bound = largest_particle(atoms.positions)
        selected = bound & (np.abs(centred[:, 2]) < 2.0) if central_slice else np.ones(len(atoms), dtype=bool)
        view = atoms[selected]
        view.positions = centred[selected]
        if len(view):
            options = {"colors": np.asarray(colours)[selected].tolist()} if colours is not None else {}
            if central_slice:
                extent = float(np.max(np.abs(centred[bound, :2]))) + 1.0
                options["bbox"] = (-extent, -extent, extent, extent)
            plot_atoms(view, axes, rotation="0x,0y,0z" if central_slice else "15x,20y,0z",
                       show_unit_cell=0, radii=0.55, **options)
        axes.set_aspect("equal")
        axes.set_axis_off()
        return np.array([extent, extent]) if central_slice and len(view) else np.zeros(2)

    return draw_particle, frame_label, largest_particle


@app.cell(hide_code=True)
def _(ConvexHull, QhullError, cKDTree, largest_particle, np):
    def radial_geometry(positions):
        centred, bound = largest_particle(positions)
        positions_main = centred[bound]
        if len(positions_main) < 4:
            raise ValueError("At least four non-coplanar atoms are needed for a radial distribution.")
        try:
            hull = ConvexHull(positions_main)
        except QhullError as error:
            raise ValueError("The particle is too flat or too small for a three-dimensional radial distribution.") from error
        depth = -np.max(positions_main @ hull.equations[:, :3].T + hull.equations[:, 3], axis=1)
        return positions_main, depth, len(positions_main) / hull.volume

    def pair_distribution(positions, r_max, dr=0.05):
        """Interior-centred radial distribution, with an approximate finite-particle density."""
        positions_main, depth, density = radial_geometry(positions)
        centres = positions_main[depth > r_max]
        if len(centres) == 0 or r_max <= dr:
            raise ValueError("No interior atoms remain at this radial cutoff.")
        distances = cKDTree(centres).sparse_distance_matrix(
            cKDTree(positions_main), r_max, output_type="ndarray")["v"]
        edges = np.linspace(0.0, r_max, max(2, int(np.ceil(r_max / dr)) + 1))
        counts, _ = np.histogram(distances[distances > 1e-10], bins=edges)
        shell_volume = 4 * np.pi / 3 * (edges[1:]**3 - edges[:-1]**3)
        return 0.5 * (edges[1:] + edges[:-1]), counts / (len(centres) * density * shell_volume)

    return pair_distribution, radial_geometry


@app.cell(hide_code=True)
def _(cKDTree, np, sph_harm_y):
    def solid_like(positions, r_cut=3.05, agreement=0.7, bonds_min=6):
        """Solid-like when q6 bond directions agree with at least bonds_min neighbours."""
        count = len(positions)
        pairs = cKDTree(positions).query_pairs(r_cut, output_type="ndarray")
        if not len(pairs):
            return np.zeros(count, dtype=bool)
        first, second = pairs[:, 0], pairs[:, 1]
        bonds = positions[second] - positions[first]
        lengths = np.linalg.norm(bonds, axis=1)
        if np.any(lengths < 1e-10):
            raise ValueError("Two atoms occupy the same position.")
        polar = np.arccos(np.clip(bonds[:, 2] / lengths, -1, 1))
        azimuth = np.arctan2(bonds[:, 1], bonds[:, 0])
        harmonics = np.stack([sph_harm_y(6, order, polar, azimuth) for order in range(-6, 7)], axis=1)
        q6 = np.zeros((count, 13), complex)
        np.add.at(q6, first, harmonics)
        np.add.at(q6, second, harmonics)
        q6 /= np.maximum(np.linalg.norm(q6, axis=1, keepdims=True), 1e-12)
        agrees = np.real(np.sum(q6[first] * np.conj(q6[second]), axis=1)) > agreement
        aligned = np.bincount(first[agrees], minlength=count) + np.bincount(second[agrees], minlength=count)
        return aligned >= bonds_min

    return (solid_like,)


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ase==3.29.0",
#     "marimo==0.25.2",
#     "matplotlib==3.11.2",
#     "numpy==2.5.4",
#     "openmm==8.6.1",
#     "plotly==7.1.0",
#     "scipy==1.18.1",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Build your own potential for an MD simulation

    In the MD simulations, we used an embedded-atom method (EAM) potential to simulate melting
    and cooling of copper nanoparticles. How was that potential made? In this notebook, you
    will see how to build one yourself.

    1. Read and plot the Mishin potential and connect its functions to OpenMM.
    2. Follow the density and energy contributions around one atom.
    3. Inspect the supplied DFT structures, energies, and forces.
    4. Fit a Lennard–Jones model, then build and fit an EAM model.
    5. Check the fitted model and enter its 15 coefficients in the OpenMM function skeleton.
    6. Copy your completed function into **notebook 1** and select **Custom**.

    The input files are supplied. Work in your own editable molab copy.
    """)
    return


@app.cell(hide_code=True)
def imports():
    import io
    import json
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import openmm as mm
    import openmm.unit as unit
    from ase import Atoms
    from ase.build import bulk
    from ase.io import read
    from ase.neighborlist import neighbor_list
    from ase.visualize.plot import plot_atoms

    return (
        Atoms,
        Path,
        bulk,
        go,
        json,
        mm,
        mo,
        neighbor_list,
        np,
        plot_atoms,
        plt,
        read,
        unit,
    )


@app.cell(hide_code=True)
def provided_files(Path, mo):
    notebook_directory = Path(mo.notebook_dir() or Path.cwd()).resolve()
    input_directories = (notebook_directory, notebook_directory.parent / "project" / "data")
    potential_path = notebook_directory / "Cu_mishin1.eam.alloy"
    fit_path = next((root / "cu_vasp_train.xyz" for root in input_directories if (root / "cu_vasp_train.xyz").is_file()), None)
    heldout_path = next((root / "cu_vasp_test.xyz" for root in input_directories if (root / "cu_vasp_test.xyz").is_file()), None)
    atomic_reference_path = next((root / "cu_vasp_atom_reference.json" for root in input_directories if (root / "cu_vasp_atom_reference.json").is_file()), None)
    missing_inputs = [name for name, available in (
        ("Cu_mishin1.eam.alloy", potential_path.is_file()),
        ("cu_vasp_train.xyz", fit_path is not None),
        ("cu_vasp_test.xyz", heldout_path is not None),
        ("cu_vasp_atom_reference.json", atomic_reference_path is not None)) if not available]
    mo.stop(bool(missing_inputs), mo.md("The instructor's notebook package is missing: " + ", ".join(missing_inputs)))
    return atomic_reference_path, fit_path, heldout_path, potential_path


@app.cell(hide_code=True)
def reference_intro(mo):
    mo.md(r"""
    ## 1 · Read an existing EAM potential

    The supplied file is the Cu EAM1 potential of
    [Mishin and co-workers (2001)](https://doi.org/10.1103/PhysRevB.63.224106).
    Its energy uses three functions:

    - **Pair energy \(\phi(r_{ij})\):** the interaction between atoms separated by \(r_{ij}\).
    - **Neighbour-density contribution \(\rho(r)\):** one neighbour's contribution at distance \(r\).
    - **Embedding energy \(F(\rho_i)\):** the energy associated with atom \(i\)'s total local density.

    For atom 1, add the neighbour contributions first:
    \(\rho_1=\rho(r_{12})+\rho(r_{13})+\cdots\). Then evaluate \(F(\rho_1)\).
    For the whole structure,

    \[
    E=\sum_{i<j}\phi(r_{ij})+\sum_i F(\rho_i),
    \qquad \rho_i=\sum_{j\ne i}\rho(r_{ij}).
    \]

    Here \(\rho(r)\) is one contribution and \(\rho_i\) is their sum. The code stores the
    neighbour-density function in the entry named density. It has a distance grid,
    while the embedding function has a separate total-density grid.
    """)
    return


@app.cell(hide_code=True)
def read_eam(Path, np, potential_path):
    def read_setfl(path):
        """Read the 3 tables and their grids from a single-element setfl file."""
        lines = Path(path).read_text().splitlines()
        nrho, drho, nr, dr, cutoff = lines[4].split()
        nrho, nr = int(nrho), int(nr)
        drho, dr, cutoff = float(drho), float(dr), float(cutoff)
        values = np.array(" ".join(lines[6:]).split(), dtype=float)
        r_grid = np.arange(nr) * dr
        rho_grid = np.arange(nrho) * drho
        embedding = values[:nrho]
        density = values[nrho:nrho + nr]
        r_phi = values[nrho + nr:nrho + 2 * nr]
        pair = np.empty(nr)
        pair[1:] = r_phi[1:] / r_grid[1:]  # setfl stores r * phi(r)
        pair[0] = pair[1]
        return dict(phi=pair, density=density, embed=embedding,
                    r=r_grid, rho=rho_grid, r_max=r_grid[-1], rho_max=rho_grid[-1], cutoff=cutoff)

    eam = read_setfl(potential_path)

    def numpy_eam(atoms, tables):
        """Energy of a free cluster, with a separate energy contribution for each atom."""
        distances = np.linalg.norm(atoms.positions[:, None, :] - atoms.positions[None, :, :], axis=2)
        neighbours = (distances > 0) & (distances < tables["cutoff"])
        density_terms = np.where(neighbours, np.interp(distances, tables["r"], tables["density"]), 0.0)
        pair_terms = np.where(neighbours, np.interp(distances, tables["r"], tables["phi"]), 0.0)
        local_density = density_terms.sum(axis=1)
        embedding = np.interp(local_density, tables["rho"], tables["embed"])
        pair_share = 0.5 * pair_terms.sum(axis=1)  # give half of each pair energy to each atom
        atomic_energy = embedding + pair_share
        return dict(distances=distances, density_terms=density_terms, pair_terms=pair_terms,
                    rho=local_density, embedding=embedding, pair_share=pair_share,
                    atomic_energy=atomic_energy, total=atomic_energy.sum())

    return eam, numpy_eam


@app.cell(hide_code=True)
def reference_curves(eam, plt):
    reference_figure, reference_axes = plt.subplots(1, 3, figsize=(11, 3.2))
    reference_distance = (eam["r"] >= 1.8) & (eam["r"] <= eam["cutoff"])
    reference_axes[0].plot(eam["r"][reference_distance], eam["phi"][reference_distance], color="tab:red")
    reference_axes[0].set(xlabel="Pair distance r (Å)", ylabel="Pair energy φ(r) (eV)")
    reference_axes[1].plot(eam["r"][reference_distance], eam["density"][reference_distance], color="tab:blue")
    reference_axes[1].set(xlabel="Neighbour distance r (Å)", ylabel="Density contribution ρ(r)")
    reference_axes[2].plot(eam["rho"], eam["embed"], color="tab:green")
    reference_axes[2].set(xlabel="Total local density ρ", ylabel="Embedding energy F(ρ) (eV)")
    reference_axes[0].set_title("Pair energy φ(rᵢⱼ)")
    reference_axes[1].set_title("Neighbour density ρ(r)")
    reference_axes[2].set_title("Embedding energy F(ρ)")
    _rho_limits = reference_axes[2].get_xlim()
    _rho_minimum = float(eam["rho"][eam["embed"].argmin()])
    reference_axes[2].axvspan(
        _rho_limits[0], _rho_minimum, color="#c5e1a5", alpha=0.5, zorder=0,
    )
    reference_axes[2].axvspan(
        _rho_minimum, _rho_limits[1], color="#ffccbc", alpha=0.5, zorder=0,
    )
    reference_axes[2].set_xlim(_rho_limits)
    reference_axes[2].text(
        (_rho_limits[0] + _rho_minimum) / 2, 0.97, "More neighbours\n→ more stable",
        transform=reference_axes[2].get_xaxis_transform(), fontsize=8, ha="center", va="top",
    )
    reference_axes[2].text(
        (_rho_minimum + _rho_limits[1]) / 2, 0.97, "Too crowded\n→ less stable",
        transform=reference_axes[2].get_xaxis_transform(), fontsize=8, ha="center", va="top",
    )
    reference_figure.tight_layout()
    reference_figure
    return


@app.cell(hide_code=True)
def interpolation_intro(mo):
    mo.md(r"""
    ### Why is the EAM file so big?

    The file stores many sampled values of the three functions shown above. During a simulation,
    an interatomic distance usually falls between two entries. Interpolation supplies a function
    value between those entries. The embedding energy is interpolated on its total-density grid.
    The table reader below converts the file into arrays for these calculations.
    """)
    return


@app.cell(hide_code=True)
def openmm_intro(mo):
    mo.md(r"""
    ## 2 · Use the EAM potential in OpenMM

    Our MD simulations use **OpenMM**, which lets us construct a custom potential in Python.
    Other MD engines, such as **LAMMPS**, provide their own potential interfaces.
    The common calculation is the potential energy and its position derivatives, which give
    the forces used to advance the atoms. Here we use the same OpenMM construction as notebook 1.

    The two functions below build the system and evaluate its energy and forces:

    1. The neighbour-density function contributes to each atom's total density.
    2. The embedding function contributes one energy per atom.
    3. The pair function contributes one energy per pair.

    OpenMM uses **nm** and **kJ/mol**. We divide distances in Å by 10 and multiply energies
    in eV by 96.48533212. The neighbour-density values keep their original scale.
    """)
    return


@app.cell(hide_code=True)
def openmm_builder(mm, unit):
    def system_from_tables(tables, n_atoms):
        system = mm.System()
        for atom in range(n_atoms):
            system.addParticle(63.546)
        force = mm.CustomGBForce()
        force.addTabulatedFunction("fden", mm.Continuous1DFunction(tables["density"], 0.0, tables["r_max"] / 10))
        force.addTabulatedFunction("phi", mm.Continuous1DFunction(tables["phi"] * 96.48533212, 0.0, tables["r_max"] / 10))
        force.addTabulatedFunction("Femb", mm.Continuous1DFunction(tables["embed"] * 96.48533212, 0.0, tables["rho_max"]))
        force.addComputedValue("rho", "fden(r)", mm.CustomGBForce.ParticlePairNoExclusions)
        force.addEnergyTerm("Femb(rho)", mm.CustomGBForce.SingleParticle)
        force.addEnergyTerm("phi(r)", mm.CustomGBForce.ParticlePairNoExclusions)
        for atom in range(n_atoms):
            force.addParticle([])
        force.setNonbondedMethod(mm.CustomGBForce.CutoffNonPeriodic)
        force.setCutoffDistance(tables["cutoff"] / 10)
        system.addForce(force)
        return system


    def openmm_energy_forces(system, positions_A):
        integrator = mm.VerletIntegrator(0.002)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"), {"Threads": "2"})
        context.setPositions(positions_A / 10)
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / 96.48533212
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer) / (96.48533212 * 10)
        del context
        return energy, forces

    return openmm_energy_forces, system_from_tables


@app.cell(hide_code=True)
def atom_intro(mo):
    mo.md(r"""
    ## 3 · Follow the energy of one atom

    Place 4 Cu atoms on a line. **Click an atom** to choose it, and move the spacing slider to change
    its neighbours' distances. Atom 2 is selected initially.

    For the chosen atom, add the density contributions first and then evaluate its embedding energy.
    To display an energy for each atom, assign **half of each pair energy to each endpoint**:

    $$E_i=F(\rho_i)+\tfrac12\sum_{j\ne i}\phi(r_{ij}).$$

    Adding the 4 atomic energies counts every pair exactly once. The embedding term makes the model
    sensitive to the combined environment: adding a neighbour changes $F$ of the total density.
    """)
    return


@app.cell(hide_code=True)
def spacing_control(mo):
    atom_spacing = mo.ui.slider(2.2, 3.5, step=0.02, value=2.56, label="Spacing between atoms (Å)", show_value=True)
    atom_spacing
    return (atom_spacing,)


@app.cell(hide_code=True)
def atom_picker(
    Atoms,
    atom_spacing,
    get_selected_atom,
    go,
    mo,
    np,
    remember_atom,
):
    four_atoms = Atoms("Cu4", positions=np.column_stack((np.arange(4) * atom_spacing.value, np.zeros(4), np.zeros(4))))
    atom_figure = go.Figure(go.Scatter(x=four_atoms.positions[:, 0], y=np.zeros(4), mode="lines+markers+text",
        text=["Atom 1", "Atom 2", "Atom 3", "Atom 4"], textposition="top center",
        marker=dict(size=38, color="#b87333", line=dict(color="#73401f", width=2)),
        line=dict(color="#999999", width=1), selectedpoints=[get_selected_atom()],
        selected=dict(marker=dict(opacity=1), textfont=dict(color="#222222")), unselected=dict(marker=dict(opacity=0.7), textfont=dict(color="#333333")),
        hovertemplate="%{text}<extra></extra>"))
    atom_figure.update_layout(height=210, margin=dict(l=35, r=35, t=40, b=40), clickmode="event+select",
        dragmode="select", showlegend=False, xaxis=dict(title="Position (Å)", fixedrange=True),
        yaxis=dict(visible=False, range=[-1, 1], fixedrange=True))
    atom_picker = mo.ui.plotly(atom_figure, config={"displayModeBar": False, "scrollZoom": False}, on_change=remember_atom)
    atom_picker
    return (four_atoms,)


@app.cell(hide_code=True)
def selected_atom(eam, four_atoms, get_selected_atom, mo, np, numpy_eam, plt):
    selected_atom = get_selected_atom()
    four_energy = numpy_eam(four_atoms, eam)
    neighbour_table = "\n".join(
        f"| {j + 1} | {four_energy['distances'][selected_atom, j]:.3f} | "
        f"{four_energy['density_terms'][selected_atom, j]:.5f} | {four_energy['pair_terms'][selected_atom, j]:.5f} | "
        f"{0.5 * four_energy['pair_terms'][selected_atom, j]:.5f} |"
        for j in range(4) if j != selected_atom)
    _atom_numbers = mo.md(f"""
    ### Atom {selected_atom + 1}

    | Neighbour | Distance (Å) | Density contribution ρ(r) | Pair energy φ(r) (eV) | Half-pair share (eV) |
    |:--|--:|--:|--:|--:|
    {neighbour_table}

    - Total local density: **{four_energy['rho'][selected_atom]:.5f}**.
    - Embedding energy F(ρ): **{four_energy['embedding'][selected_atom]:.5f} eV**.
    - Sum of half-pair shares: **{four_energy['pair_share'][selected_atom]:.5f} eV**.
    - This atom's assigned energy: **{four_energy['atomic_energy'][selected_atom]:.5f} eV**.
    - Total energy of all 4 atoms: **{four_energy['total']:.5f} eV**.
    """)

    selected_neighbours = np.arange(4) != selected_atom
    selected_distances = four_energy["distances"][selected_atom, selected_neighbours]
    curve_distances = np.linspace(1.8, max(6.0, selected_distances.max() + 0.2), 500)
    curve_inside = curve_distances < eam["cutoff"]
    atom_curve_figure, atom_curve_axes = plt.subplots(1, 3, figsize=(11, 3.2))
    atom_curve_axes[0].plot(curve_distances, np.where(curve_inside, np.interp(curve_distances, eam["r"], eam["phi"]), 0), color="tab:red")
    atom_curve_axes[0].scatter(selected_distances, four_energy["pair_terms"][selected_atom, selected_neighbours], color="black", zorder=3)
    atom_curve_axes[0].set(xlabel="Neighbour distance (Å)", ylabel="Pair energy φ(r) (eV)")
    atom_curve_axes[1].plot(curve_distances, np.where(curve_inside, np.interp(curve_distances, eam["r"], eam["density"]), 0), color="tab:blue")
    atom_curve_axes[1].scatter(selected_distances, four_energy["density_terms"][selected_atom, selected_neighbours], color="black", zorder=3)
    atom_curve_axes[1].set(xlabel="Neighbour distance (Å)", ylabel="Density contribution ρ(r)")
    atom_curve_axes[2].plot(eam["rho"], eam["embed"], color="tab:green")
    atom_curve_axes[2].scatter(four_energy["rho"][selected_atom], four_energy["embedding"][selected_atom], color="black", zorder=3)
    atom_curve_axes[2].set(xlabel="Total local density ρ", ylabel="Embedding energy F(ρ) (eV)")
    atom_curve_figure.suptitle(f"Atom {selected_atom + 1}: black points are the values used in its energy")
    atom_curve_axes[0].set_title("Pair energy φ(rᵢⱼ)")
    atom_curve_axes[1].set_title("Neighbour density ρ(r)")
    atom_curve_axes[2].set_title("Embedding energy F(ρ)")
    atom_curve_figure.tight_layout()
    mo.vstack([_atom_numbers,atom_curve_figure])
    return (four_energy,)


@app.cell(hide_code=True)
def reference_check(
    eam,
    four_atoms,
    four_energy,
    mo,
    openmm_energy_forces,
    system_from_tables,
):
    reference_openmm_energy, reference_openmm_forces = openmm_energy_forces(system_from_tables(eam, 4), four_atoms.positions)
    mo.md(f"**Same 4-atom structure:** NumPy energy = {four_energy['total']:.7f} eV; "
          f"OpenMM energy = {reference_openmm_energy:.7f} eV. "
          f"The interpolation difference is {abs(reference_openmm_energy - four_energy['total']):.2g} eV.")
    return


@app.cell(hide_code=True)
def data_intro(mo):
    mo.md(r"""
    ## 4 · Inspect the supplied DFT calculations

    Each supplied XYZ frame contains a small Cu structure and a completed VASP calculation:

    - atomic positions and the periodic cell, in Å;
    - the cell's total DFT energy, `dft_energy`, in eV;
    - the 3 force components on each atom, `dft_forces`, in eV/Å;
    - a structure label: distorted fcc, bcc, hcp, or a (111) slab with a surface.

    The supplied calculations use PBE PAW, a 400 eV plane-wave cutoff, a $4\times4\times4$ bulk
    k-point mesh or $4\times4\times1$ slab mesh, and 0.15 eV smearing. The energies are VASP's
    finite-smearing free energies, consistent with the force labels.

    **The files are already separated:** fit using `cu_vasp_train.xyz`; check the result using
    `cu_vasp_test.xyz`. Both are supplied. Move the structure slider to inspect the data.
    """)
    return


@app.cell
def read_dft(fit_path, heldout_path, mo, read):
    fit_structures = read(fit_path, index=":", format="extxyz")
    heldout_structures = read(heldout_path, index=":", format="extxyz")
    mo.md(f"**Provided:** {len(fit_structures)} fitting structures and {len(heldout_structures)} held-out structures.")
    return fit_structures, heldout_structures


@app.cell(hide_code=True)
def dft_file(mo):
    dft_file = mo.ui.dropdown({"Fitting file": "fit", "Held-out file": "heldout"}, value="Fitting file", label="Inspect")
    dft_file
    return (dft_file,)


@app.cell(hide_code=True)
def dft_frame(dft_file, fit_structures, heldout_structures, mo):
    inspected_structures = fit_structures if dft_file.value == "fit" else heldout_structures
    dft_frame = mo.ui.slider(0, len(inspected_structures)-1, value=0, label="DFT structure", show_value=True)
    dft_frame
    return dft_frame, inspected_structures


@app.cell(hide_code=True)
def dft_view(
    dft_atom_energy,
    dft_frame,
    inspected_structures,
    mo,
    np,
    plot_atoms,
    plt,
):
    dft_atoms = inspected_structures[dft_frame.value]
    dft_picture = dft_atoms.repeat((2, 2, 1))
    dft_figure, dft_axis = plt.subplots(figsize=(4.2, 3.6))
    plot_atoms(dft_picture, dft_axis, rotation="20x,25y,0z", show_unit_cell=2, radii=0.65)
    dft_axis.set_axis_off()
    dft_figure.tight_layout()
    mo.hstack([dft_figure, mo.md(f"""
    **{dft_atoms.info.get('family', 'Cu')} · structure {dft_frame.value}**

    - Original cell: {len(dft_atoms)} atoms.
    - DFT cell energy: {dft_atoms.info['dft_energy']:.6f} eV.
    - DFT energy per atom: {dft_atoms.info['dft_energy']/len(dft_atoms):.6f} eV/atom.
    - Interaction energy per atom, relative to isolated Cu: **{dft_atoms.info['dft_energy']/len(dft_atoms)-dft_atom_energy:.6f} eV/atom**.
    - Largest force magnitude: {np.linalg.norm(dft_atoms.arrays['dft_forces'], axis=1).max():.3f} eV/Å.

    The picture repeats the original cell twice in x and y.
    """)], justify="start", align="center", wrap=True)
    return


@app.cell(hide_code=True)
def fit_targets(dft_atom_energy, fit_structures, np):
    fit_sizes = np.array([len(atoms) for atoms in fit_structures])
    energy_scale, force_scale, ridge_penalty = 0.03, 0.05, 1e-5
    _reference_energy = np.array([a.info["dft_energy"]-len(a)*dft_atom_energy for a in fit_structures])
    _reference_force = [a.arrays["dft_forces"] for a in fit_structures]
    b = np.concatenate([_reference_energy/(fit_sizes*energy_scale),
                        *[f.ravel()/(force_scale*np.sqrt(3*len(f))) for f in _reference_force]])
    return b, energy_scale, fit_sizes, force_scale, ridge_penalty


@app.cell(hide_code=True)
def lj_intro(mo):
    mo.md(r"""
    ## 5 · How do we create our own model? Start with Lennard–Jones

    The Lennard–Jones (LJ) model adds the same pair function over all atom pairs:

    $$E_\mathrm{LJ}=\sum_{i<j}\phi(r_{ij}),\qquad
    \phi(r)=4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right].$$

    - **$\epsilon$ (eV):** depth of the pair-energy minimum.
    - **$\sigma$ (Å):** distance where the unshifted pair energy is zero. Its minimum is at $2^{1/6}\sigma$.
    - **Cutoff:** 5.5 Å, as in the fitted EAM model. We use $\phi(r)-\phi(r_c)$ below the cutoff
      and zero above it. This shift makes the energy continuous and leaves the force inside the cutoff unchanged.

    We fit both positive parameters to the same DFT training energies and forces, using the same fixed
    atomic reference, energy and force scaling, column normalization, and ridge penalty used below for EAM.
    Writing $\phi(r)=C_{12}r^{-12}-C_6r^{-6}$ makes the fit linear in the 2 positive coefficients:
    $C_{12}=4\epsilon\sigma^{12}$ and $C_6=4\epsilon\sigma^6$.
    The held-out structures are used only for the comparison after fitting.
    """)
    return


@app.cell(hide_code=True)
def lj_features(cutoff, neighbor_list, np):
    def lj_feature_rows(atoms):
        """Energy and force features for energy-shifted LJ, including periodic images."""
        i, j, vectors, r = neighbor_list("ijDd", atoms, cutoff, self_interaction=False)
        pair_basis = np.column_stack((r**-12 - cutoff**-12, -(r**-6 - cutoff**-6)))
        radial_derivative = np.column_stack((-12 * r**-13, 6 * r**-7))
        forces = np.zeros((len(atoms), 3, 2))
        # The neighbour list includes both directions: halve energy, sum force at each centre.
        np.add.at(forces, i, (vectors / r[:, None])[:, :, None] * radial_derivative[:, None, :])
        return 0.5 * pair_basis.sum(axis=0), forces

    return (lj_feature_rows,)


@app.cell(hide_code=True)
def lj_fit(
    b,
    energy_scale,
    fit_sizes,
    fit_structures,
    force_scale,
    lj_feature_rows,
    mo,
    np,
    ridge_penalty,
):
    from scipy.optimize import nnls

    lj_training_rows = [lj_feature_rows(atoms) for atoms in fit_structures]
    lj_matrix = np.vstack([
        np.array([row[0] for row in lj_training_rows]) / (fit_sizes[:, None] * energy_scale),
        *[row[1].reshape(-1, 2) / (force_scale * np.sqrt(3 * len(row[1]))) for row in lj_training_rows],
    ])
    lj_column_scale = np.linalg.norm(lj_matrix, axis=0)
    lj_coefficients = nnls(
        np.vstack([lj_matrix / lj_column_scale, np.sqrt(ridge_penalty) * np.eye(2)]),
        np.r_[b, np.zeros(2)],
    )[0] / lj_column_scale
    mo.stop(np.any(lj_coefficients <= 0), mo.md("The fit reached a zero-coefficient boundary; a finite LJ minimum requires both coefficients to be positive."))
    lj_epsilon = lj_coefficients[1]**2 / (4 * lj_coefficients[0])
    lj_sigma = (lj_coefficients[0] / lj_coefficients[1])**(1/6)
    mo.md(f"**Fitted LJ parameters:** ε = **{lj_epsilon:.6f} eV**, σ = **{lj_sigma:.6f} Å**. "
          "Both parameters are fitted jointly to the training energies and forces.")
    return (lj_coefficients,)


@app.cell(hide_code=True)
def lj_only_validation(
    dft_atom_energy,
    heldout_structures,
    lj_coefficients,
    lj_feature_rows,
    mo,
    np,
    plt,
):
    _sizes = np.array([len(a) for a in heldout_structures])
    _energy = np.array([a.info["dft_energy"]/len(a)-dft_atom_energy for a in heldout_structures])
    _force = np.concatenate([a.arrays["dft_forces"].ravel() for a in heldout_structures])
    _rows = [lj_feature_rows(a) for a in heldout_structures]
    _pred_energy = np.array([row[0]@lj_coefficients for row in _rows])/_sizes
    _pred_force = np.concatenate([(row[1]@lj_coefficients).ravel() for row in _rows])
    _figure,_axes = plt.subplots(1,2,figsize=(9,3.6))
    for _ax,_reference,_predicted,_label in ((_axes[0],_energy,_pred_energy,"interaction energy (eV/atom)"),
                                            (_axes[1],_force,_pred_force,"force (eV/Å)")):
        _ax.scatter(_reference,_predicted,s=12,alpha=0.5,color="tab:orange",label="Lennard–Jones")
        _limits = [min(_reference.min(),_predicted.min()),max(_reference.max(),_predicted.max())]
        _ax.plot(_limits,_limits,"k--",lw=1)
        _ax.set(xlabel="DFT "+_label,ylabel="LJ "+_label)
        _ax.legend(frameon=False)
    _figure.tight_layout()
    mo.vstack([mo.md(f"""
    ### How well does the LJ model predict the held-out structures?

    - Energy mean absolute error: **{np.mean(np.abs(_pred_energy-_energy))*1000:.2f} meV/atom**.
    - Force-component mean absolute error: **{np.mean(np.abs(_pred_force-_force))*1000:.2f} meV/Å**.

    The dashed line marks agreement with DFT. The spread shows where this fitted pair model
    misses the energies and forces. Next we add a dependence on the combined atomic environment.
    """),_figure])
    return


@app.cell(hide_code=True)
def custom_intro(mo):
    mo.md(r"""
    ## 6 · Build functions with linear coefficients

    The LJ fit uses two parameters for all atomic environments. We now add an embedding energy
    that depends on the combined density around each atom. We keep the neighbour-density function
    fixed and fit the pair and embedding functions. The cutoff is \(r_c=5.5\) Å:

    \[
    \rho(r)=\left[\frac{\max(5.5-r,0)}{5.5}\right]^4.
    \]

    ### The cubic building blocks

    Each building block is a curve that falls to zero at its own fixed distance \(t_k\), called a **knot**:

    \[
    B_k(r)=\max(t_k-r,0)^3=
    \begin{cases}(t_k-r)^3,&r<t_k,\\0,&r\ge t_k.\end{cases}
    \]

    The 12 knot distances are **2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, and 5.5 Å**.
    Every block is evaluated at the same interatomic distance \(r\), so several can contribute at once.
    For example, the block ending at 2.5 Å gives \((2.5-2.0)^3=0.125\) Å³ at \(r=2.0\) Å
    and zero at \(r=3.0\) Å. Its coefficient scales that contribution up or down.

    The pair potential is the sum of these scaled curves:
    \(\phi(r)=a_1B_1(r)+a_2B_2(r)+\cdots+a_{12}B_{12}(r)\).
    A positive coefficient raises the pair energy and a negative coefficient lowers it.
    The fit adjusts the 12 coefficients while the knot distances stay fixed.

    ### Why use 12 cutoff-based pair terms?

    Atoms in distorted crystals and near surfaces have different neighbour distances. We want the
    pair function to adjust its shape across those distances while giving smooth forces as atoms move.

    - **Flexible shape:** the 12 overlapping cubic blocks let the fit adjust the pair curve over
      a range of distances. Their spacing is 0.3 Å between 2.2 and 5.5 Å. Twelve is a manageable
      modelling choice for this exercise. The held-out errors help assess how well it fits the
      supplied structures.
    - **Smooth cutoff:** each block and its first two derivatives reach zero at its knot.
      The summed pair function therefore has continuous energy, slope, and curvature, and
      vanishes beyond the outermost knot at 5.5 Å. The pair force, obtained from the negative
      slope, goes smoothly to zero as neighbours move beyond this distance.
    - **Finite range and a linear fit:** the cutoff limits the pair calculation to nearby atoms.
      This assumes that omitted, more distant pair contributions are small enough for the
      intended calculation. With fixed knots, both the pair energies and their position
      derivatives are linear in the 12 coefficients, so energies and forces can enter the same
      least-squares fit.

    The figure below shows three building blocks separately, each with its own vertical scale.

    - **Pair function:** use 12 cubic pieces with fixed knots \(t_1,\ldots,t_{12}\) from 2.2 to 5.5 Å:

      \[
      \phi(r)=a_1\max(t_1-r,0)^3+\cdots+a_{12}\max(t_{12}-r,0)^3.
      \]

    - **Embedding function:** use three powers of the total local density:

      \[
      F(\rho_i)=c_1\sqrt{\rho_i}+c_2\rho_i^{3/2}+c_3\rho_i^2.
      \]

    These are **15 unknown coefficients**: 12 for the pair function and three for embedding.
    For known atomic positions, all distances and densities are known. The energy and force
    components are linear in these coefficients, so they can be fitted together by linear least squares.
    The interaction energy is measured from separated atoms, with \(F(0)=0\).
    """)
    return


@app.cell(hide_code=True)
def pair_basis_example(mo, np, plt):
    _r = np.linspace(1.8,5.8,500)
    _fig,_axes = plt.subplots(1,3,figsize=(10,3.2))
    for _ax,_t,_number in zip(_axes,(2.5,3.7,5.5),(2,6,12)):
        _ax.plot(_r,np.maximum(_t-_r,0)**3,color='tab:blue',lw=2)
        _ax.axvline(_t,color='gray',ls='--',lw=1)
        _ax.scatter([_t],[0],color='tab:red',zorder=3)
        _ax.set(title=f'Block {_number}: ends at {_t:g} Å',xlabel='Distance r (Å)',ylabel='Building-block value (Å³)')
    _fig.tight_layout()
    mo.vstack([mo.md('### Three of the 12 cubic building blocks'),_fig])
    return


@app.cell(hide_code=True)
def basis_curves(mo, np, plt):
    knots = np.linspace(2.2, 5.5, 12)
    cutoff = 5.5
    basis_r = np.linspace(1.8, 5.5, 400)
    basis_rho = np.linspace(0, 2.5, 400)
    basis_figure, basis_axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for index in (0, 5, 11):
        basis_axes[0].plot(basis_r, np.maximum(knots[index] - basis_r, 0)**3, label=f"t = {knots[index]:.1f} Å")
    basis_axes[0].set(xlabel="r (Å)", ylabel="Example pair basis functions")
    basis_axes[0].legend(frameon=False, fontsize=8)
    basis_axes[1].plot(basis_r, (np.maximum(5.5 - basis_r, 0) / 5.5)**4)
    basis_axes[1].set(xlabel="r (Å)", ylabel="Fixed neighbour density ρ(r)")
    for power in (0.5, 1.5, 2):
        basis_axes[2].plot(basis_rho, basis_rho**power, label=f"ρ^{power:g}")
    basis_axes[2].set(xlabel="Total local density ρ", ylabel="Embedding basis functions")
    basis_axes[2].legend(frameon=False, fontsize=8)
    basis_figure.tight_layout()
    mo.accordion({"Optional · Pair and density basis functions": basis_figure})
    return cutoff, knots


@app.cell(hide_code=True)
def energy_force_features(cutoff, knots, neighbor_list, np):
    def feature_rows(atoms):
        n = len(atoms)
        i, j, vectors, r = neighbor_list("ijDd", atoms, cutoff, self_interaction=False)
        direction = vectors / r[:, None]
        x = np.maximum(knots[None, :] - r[:, None], 0.0)
        pair = x**3
        pair_derivative = -3.0 * x**2
        density_pair = np.maximum(cutoff - r, 0.0)**4 / cutoff**4
        density_derivative = -4.0 * np.maximum(cutoff - r, 0.0)**3 / cutoff**4
        rho = np.bincount(i, weights=density_pair, minlength=n)
        root = np.sqrt(rho)
        embedding = np.column_stack((root, rho * root, rho**2))
        energy = np.r_[0.5 * pair.sum(axis=0), embedding.sum(axis=0)]
        embed_derivative = np.column_stack((0.5 / np.maximum(root, 1e-12), 1.5 * root, 2 * rho))
        radial = np.column_stack((0.5 * pair_derivative,
                                  embed_derivative[i] * density_derivative[:, None]))
        contribution = direction[:, :, None] * radial[:, None, :]
        forces = np.zeros((n, 3, len(energy)))
        np.add.at(forces, i, contribution)
        np.add.at(forces, j, -contribution)
        return energy, forces

    def build_matrix(structures):
        energies, forces = zip(*(feature_rows(atoms) for atoms in structures))
        return np.array(energies), list(forces)

    return build_matrix, feature_rows


@app.cell(hide_code=True)
def linear_system_intro(mo):
    mo.md(r"""
    ## 7 · Fit the EAM coefficients and check the held-out file

    For each structure, the first 12 features add the cubic pair functions over all pairs.
    The last three add \(\sqrt{\rho_i}\), \(\rho_i^{3/2}\), and \(\rho_i^2\) over all atoms.
    Writing these known quantities as \(A_1,\ldots,A_{15}\) gives

    \[
    E_\mathrm{int}\approx A_1a_1+\cdots+A_{12}a_{12}+A_{13}c_1+A_{14}c_2+A_{15}c_3.
    \]

    Negative position derivatives give corresponding equations for the force components.
    Stacking them produces \(A\boldsymbol{w}\approx\boldsymbol{b}\).
    The neighbour list includes periodic images across cell boundaries.

    We use the same energy and force targets, scaling, and small ridge penalty as the LJ fit.
    The held-out file is used after fitting to measure prediction errors.
    """)
    return


@app.cell(hide_code=True)
def matrices(
    build_matrix,
    energy_scale,
    fit_sizes,
    fit_structures,
    force_scale,
    heldout_structures,
    mo,
    np,
):
    energy_fit, force_fit = build_matrix(fit_structures)
    energy_heldout, force_heldout = build_matrix(heldout_structures)
    A = np.vstack([energy_fit / (fit_sizes[:, None] * energy_scale),
                   *[f.reshape(-1,15)/(force_scale*np.sqrt(3*len(f))) for f in force_fit]])
    mo.md(f"The fitting system has **{A.shape[0]} equations** and **{A.shape[1]} unknown coefficients**.")
    return A, energy_heldout, force_heldout


@app.cell(hide_code=True)
def fitting_helpers(A, b, np, ridge_penalty):
    # Normalize the columns so coefficients with different scales can be fitted together.
    column_scale = np.linalg.norm(A, axis=0)
    scaled_A = A / column_scale


    def fitting_objective(scaled_coefficients):
        residual = scaled_A @ scaled_coefficients - b
        return 0.5 * (
            residual @ residual + ridge_penalty * (scaled_coefficients @ scaled_coefficients)
        )


    def fitting_gradient(scaled_coefficients):
        residual = scaled_A @ scaled_coefficients - b
        return scaled_A.T @ residual + ridge_penalty * scaled_coefficients

    return column_scale, fitting_gradient, fitting_objective, scaled_A


@app.cell
def fit(column_scale, fitting_gradient, fitting_objective, mo, np, scaled_A):
    #################################
    ## The actual fitting code is here
    ## Use scipy.optimize.minimize to fit the 15 coefficients.
    #################################
    from scipy.optimize import minimize

    initial_guess = np.zeros(scaled_A.shape[1])
    fit_result = minimize(
        fitting_objective,
        initial_guess,
        jac=fitting_gradient,
        method="BFGS",
        options={"gtol": 1e-7, "maxiter": 1000},
    )
    if not fit_result.success:
        raise RuntimeError(f"Fit did not converge: {fit_result.message}")

    coefficients = fit_result.x / column_scale
    mo.md(f"**Fit converged:** {fit_result.nit} iterations.")

    return (coefficients,)


@app.cell(hide_code=True)
def heldout(
    coefficients,
    dft_atom_energy,
    energy_heldout,
    force_heldout,
    heldout_structures,
    lj_coefficients,
    lj_feature_rows,
    mo,
    np,
    plt,
):
    heldout_sizes = np.array([len(atoms) for atoms in heldout_structures])
    heldout_reference_energy = np.array([atoms.info["dft_energy"] / len(atoms) - dft_atom_energy for atoms in heldout_structures])
    heldout_reference_force = np.concatenate([atoms.arrays["dft_forces"].ravel() for atoms in heldout_structures])
    heldout_predicted_energy = energy_heldout @ coefficients / heldout_sizes
    heldout_predicted_force = np.concatenate([(features @ coefficients).ravel() for features in force_heldout])
    lj_heldout_rows = [lj_feature_rows(atoms) for atoms in heldout_structures]
    lj_predicted_energy = np.array([row[0] @ lj_coefficients for row in lj_heldout_rows]) / heldout_sizes
    lj_predicted_force = np.concatenate([(row[1] @ lj_coefficients).ravel() for row in lj_heldout_rows])
    energy_mae = np.mean(np.abs(heldout_predicted_energy - heldout_reference_energy))
    force_mae = np.mean(np.abs(heldout_predicted_force - heldout_reference_force))
    lj_energy_mae = np.mean(np.abs(lj_predicted_energy - heldout_reference_energy))
    lj_force_mae = np.mean(np.abs(lj_predicted_force - heldout_reference_force))
    parity_figure, parity_axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, reference, predicted, lj_predicted, label in (
        (parity_axes[0], heldout_reference_energy, heldout_predicted_energy, lj_predicted_energy, "interaction energy (eV/atom)"),
        (parity_axes[1], heldout_reference_force, heldout_predicted_force, lj_predicted_force, "force (eV/Å)")):
        axis.scatter(reference, lj_predicted, s=12, alpha=0.4, marker="x", color="tab:orange", label="Lennard–Jones")
        axis.scatter(reference, predicted, s=9, alpha=0.45, color="tab:blue", label="Fitted EAM")
        limits = [min(reference.min(), predicted.min(), lj_predicted.min()),
                  max(reference.max(), predicted.max(), lj_predicted.max())]
        axis.plot(limits, limits, "k--", lw=1)
        axis.set(xlabel="DFT " + label, ylabel="Predicted " + label)
        axis.legend(frameon=False, fontsize=8)
    parity_figure.tight_layout()
    mo.vstack([mo.md(f"""
    ### Held-out energy and force errors

    | Model | Numbers fitted | Energy MAE (meV/atom) | Force-component MAE (meV/Å) |
    |:--|--:|--:|--:|
    | Lennard–Jones | 2 | {lj_energy_mae*1000:.2f} | {lj_force_mae*1000:.2f} |
    | Fitted EAM | 15 | {energy_mae*1000:.2f} | {force_mae*1000:.2f} |

    MAE is the mean absolute error. Energy errors use one energy per atom for each structure;
    force errors use all Cartesian force components. Points on the dashed line agree with DFT.
    The Lennard–Jones energy error is **{lj_energy_mae/energy_mae:.2f} times** the EAM error,
    and its force-component error is **{lj_force_mae/force_mae:.2f} times** the EAM error.
    The pairwise LJ model uses the same two parameters for every atomic environment.
    The EAM embedding term makes an atom's energy depend on its total neighbour density.
    """), parity_figure])
    return


@app.cell(hide_code=True)
def coefficients_table(coefficients, mo):
    coefficient_labels = [f"a{k+1}" for k in range(12)] + ["c1", "c2", "c3"]
    coefficient_rows = "\n".join(f"| {label} | {value:.10g} |" for label,value in zip(coefficient_labels, coefficients))
    _coefficient_table = mo.md("### The 15 fitted coefficients\n\n| Coefficient | Value |\n|:--|--:|\n" + coefficient_rows +
          "\n\nDistances are in Å and energies in eV. The embedding coefficients use eV "
          "with the prescribed dimensionless density, and the pair coefficients use eV/Å³.")
    _coefficient_equation = mo.md(r"""
    ### Where do the 15 coefficients enter?

    \[
    \begin{aligned}
    E={}&\sum_{i<j}\left[
     a_1\max(t_1-r_{ij},0)^3+\cdots+a_{12}\max(t_{12}-r_{ij},0)^3\right]\\
     &+\sum_i\left[c_1\sqrt{\rho_i}+c_2\rho_i^{3/2}+c_3\rho_i^2\right],
    \qquad \rho_i=\sum_{j\ne i}\rho(r_{ij}).
    \end{aligned}
    \]

    - \(a_1,\ldots,a_{12}\) weight the 12 radial cubic pieces.
    - \(c_1,c_2,c_3\) weight the three powers of total density.
    - The density function and the 12 knot positions stay fixed. There are no cross-product
      coefficients between pair and density functions.

    Enter the values in the skeleton below in this order: \(a_1,\ldots,a_{12},c_1,c_2,c_3\).
    """)
    mo.vstack([_coefficient_table,_coefficient_equation])
    return


@app.cell(hide_code=True)
def fitted_tables(coefficients, knots, mo, np, plt):
    fitted_r = np.linspace(0.0, 5.5, 10001)
    fitted_rho = np.linspace(0.0, 10.0, 20001)
    fitted_phi = np.maximum(knots[None, :] - fitted_r[:, None], 0.0)**3 @ coefficients[:12]
    fitted_density = (np.maximum(5.5 - fitted_r, 0.0) / 5.5)**4
    fitted_embedding = np.column_stack((np.sqrt(fitted_rho), fitted_rho**1.5, fitted_rho**2)) @ coefficients[12:]
    fitted_tables = dict(phi=fitted_phi, density=fitted_density, embed=fitted_embedding, r=fitted_r,
                         rho=fitted_rho, r_max=5.5, rho_max=10.0, cutoff=5.5)
    fitted_figure, fitted_axes = plt.subplots(1, 3, figsize=(11, 3.2))
    fitted_visible_r = fitted_r >= 1.8
    fitted_visible_rho = fitted_rho <= 2.5
    fitted_axes[0].plot(fitted_r[fitted_visible_r], fitted_phi[fitted_visible_r], color="tab:red")
    fitted_axes[0].set(xlabel="r (Å)", ylabel="Fitted pair energy φ(r) (eV)")
    fitted_axes[1].plot(fitted_r[fitted_visible_r], fitted_density[fitted_visible_r], color="tab:blue")
    fitted_axes[1].set(xlabel="r (Å)", ylabel="Fixed neighbour density ρ(r)")
    fitted_axes[2].plot(fitted_rho[fitted_visible_rho], fitted_embedding[fitted_visible_rho], color="tab:green")
    fitted_axes[2].set(xlabel="Total local density ρ", ylabel="Fitted embedding energy F(ρ) (eV)")
    fitted_axes[0].set_title("Pair energy φ(rᵢⱼ)")
    fitted_axes[1].set_title("Neighbour density ρ(r)")
    fitted_axes[2].set_title("Embedding energy F(ρ)")
    fitted_figure.tight_layout()
    mo.accordion({"Optional · Fitted EAM functions":fitted_figure})
    return


@app.cell(hide_code=True)
def binding_intro(
    binding_fit_minimum,
    binding_lj_minimum,
    binding_mishin_minimum,
    binding_spacings,
    fitted_cohesive_energy,
    lj_cohesive_energy,
    mishin_cohesive_energy,
    mo,
):
    mo.md(f"""
    ## 8 · How do we know the potential is good before doing any MD simulation?

    One check is the energy of a crystal relative to separated atoms:

    \\[
    E_\\mathrm{{coh}}=\\frac{{E_\\mathrm{{crystal}}}}{{N}}-E_\\mathrm{{atom}}.
    \\]

    This signed binding energy is negative for a bound crystal. Its positive magnitude,
    \\(-E_\\mathrm{{coh}}\\), is often called the cohesive energy. All three curves use separated
    atoms as zero. The table reports the depth of each minimum and its lattice parameter.

    | Model | Well depth −E_coh (eV/atom) | Lattice parameter (Å) |
    |:--|--:|--:|
    | Lennard–Jones | {lj_cohesive_energy:.4f} | {binding_spacings[binding_lj_minimum]:.4f} |
    | Fitted EAM | {fitted_cohesive_energy:.4f} | {binding_spacings[binding_fit_minimum]:.4f} |
    | Mishin EAM | {mishin_cohesive_energy:.4f} | {binding_spacings[binding_mishin_minimum]:.4f} |

    For experimental scale, [NIST's 298 K Cu atomization enthalpy](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440508&Units=SI&Mask=1)
    is about 3.50 eV/atom. The curves compare static-lattice potential energies, and their small kinks
    appear where a neighbour distance crosses a cutoff.

    Each row is the lowest point of that model's own curve. The fitted LJ minimum is deeper
    and at a smaller lattice parameter than either EAM minimum. Its pair function must describe
    all atomic environments with the same two parameters, while the EAM embedding term depends
    on the combined density of an atom's neighbours.
    """)
    return


@app.cell(hide_code=True)
def binding_check(
    bulk,
    coefficients,
    cutoff,
    eam,
    knots,
    lj_coefficients,
    neighbor_list,
    np,
    plt,
):
    def fcc_binding_curves(lattice_parameters):
        """Per-atom energies along a uniform expansion of an fcc crystal."""
        a_min = float(np.min(lattice_parameters))
        # At the smallest lattice parameter, include every neighbour needed by either model.
        # Expansion only moves these neighbours outwards, so this one list covers the scan.
        scaled_distances = neighbor_list(
            "d", bulk("Cu", "fcc", a=a_min), max(cutoff, eam["cutoff"]),
            self_interaction=False,
        ) / a_min
        distances = lattice_parameters[:, None] * scaled_distances[None, :]

        fitted_neighbours = distances < cutoff
        pair_basis = np.maximum(knots[None, None, :] - distances[:, :, None], 0.0)**3
        pair_features = 0.5 * (pair_basis * fitted_neighbours[:, :, None]).sum(axis=1)
        density = (np.maximum(cutoff - distances, 0.0) / cutoff)**4
        rho = density.sum(axis=1)
        energy_features = np.column_stack((pair_features, np.sqrt(rho), rho**1.5, rho**2))

        lj_features = 0.5 * np.column_stack((
            np.where(fitted_neighbours, distances**-12 - cutoff**-12, 0.0).sum(axis=1),
            np.where(fitted_neighbours, -(distances**-6 - cutoff**-6), 0.0).sum(axis=1),
        ))
        lj_energy = lj_features @ lj_coefficients

        mishin_neighbours = distances < eam["cutoff"]
        mishin_rho = np.where(
            mishin_neighbours, np.interp(distances, eam["r"], eam["density"]), 0.0,
        ).sum(axis=1)
        mishin_pair = 0.5 * np.where(
            mishin_neighbours, np.interp(distances, eam["r"], eam["phi"]), 0.0,
        ).sum(axis=1)
        mishin_energy = mishin_pair + np.interp(mishin_rho, eam["rho"], eam["embed"]) - eam["embed"][0]
        return energy_features, lj_energy, mishin_energy


    binding_spacings = np.r_[np.linspace(2.9, 4.1, 481), np.linspace(4.15, 8.0, 78)]
    binding_energy_features, binding_lj, binding_mishin = fcc_binding_curves(binding_spacings)
    binding_fitted = binding_energy_features @ coefficients
    binding_lj_minimum = int(np.argmin(binding_lj))
    lj_cohesive_energy = float(-binding_lj[binding_lj_minimum])
    binding_fit_minimum = int(np.argmin(binding_fitted))
    binding_mishin_minimum = int(np.argmin(binding_mishin))
    fitted_cohesive_energy = float(-binding_fitted[binding_fit_minimum])
    mishin_cohesive_energy = float(-binding_mishin[binding_mishin_minimum])
    binding_figure, binding_axis = plt.subplots(figsize=(7.5, 3.8))
    binding_axis.plot(binding_spacings, binding_lj, color="tab:orange", label="Lennard–Jones")
    binding_axis.plot(binding_spacings, binding_fitted, color="tab:blue", label="Fitted EAM")
    binding_axis.plot(binding_spacings, binding_mishin, color="tab:green", label="Mishin EAM")
    binding_axis.axhline(0, color="black", lw=0.7)
    binding_axis.set(xlabel="fcc lattice parameter a (Å)", ylabel="Energy relative to isolated Cu (eV/atom)")
    binding_axis.legend(frameon=False)
    binding_figure.tight_layout()
    binding_figure
    return (
        binding_fit_minimum,
        binding_lj_minimum,
        binding_mishin_minimum,
        binding_spacings,
        fitted_cohesive_energy,
        lj_cohesive_energy,
        mishin_cohesive_energy,
    )


@app.cell(hide_code=True)
def custom_function_intro(mo):
    mo.md(r"""
    ## 9 · Turn the fitted functions into an OpenMM system

    The pair function is tabulated for OpenMM. The fixed density function and the embedding
    formula are evaluated directly, including the square-root shape near zero density.

    Enter your **15 fitted coefficients** in the empty array in the next cell.
    Keep the order \(a_1,\ldots,a_{12},c_1,c_2,c_3\). Then run the cell and the checks below.
    The function converts Å to nm and eV to kJ/mol and returns an OpenMM System.
    A tiny density floor keeps the derivative finite at zero density.

    Copy your completed function into the marked Custom-potential cell in notebook 1.
    """)
    return


@app.cell
def copy_function(mm, mo, np):
    def custom_system(n_atoms):
        ###########################################################################
        # INSERT YOUR 15 FITTED COEFFICIENTS BELOW
        # Replace the empty [] with your coefficients from the table above.
        # Order: a1, a2, ..., a12, c1, c2, c3.
        # Run this cell and the checks below. Then copy this complete function
        # into the marked custom MD system cell in Notebook 1.
        # See the bonus instructions for the complete workflow.
        ###########################################################################
        coefficients = np.array([], dtype=float)
        if coefficients.shape != (15,) or not np.isfinite(coefficients).all():
            raise ValueError("Enter all 15 fitted coefficients in the coefficients array.")
        knots = np.linspace(2.2, 5.5, 12)
        r = np.linspace(0.0, 5.5, 10001)       # Å
        phi = np.maximum(knots[None, :] - r[:, None], 0.0)**3 @ coefficients[:12]

        system = mm.System()
        for atom in range(n_atoms):
            system.addParticle(63.546)
        force = mm.CustomGBForce()
        force.addTabulatedFunction("phi", mm.Continuous1DFunction(phi * 96.48533212, 0.0, 0.55))
        for name, value in zip(("c1", "c2", "c3"), coefficients[12:]):
            force.addGlobalParameter(name, float(value) * 96.48533212)
        # r is in nm. The density function is dimensionless.
        force.addComputedValue("rho", "(max(0.55-r,0)/0.55)^4", mm.CustomGBForce.ParticlePairNoExclusions)
        # A tiny floor keeps the derivative finite at rho=0; F(0) remains zero.
        force.addEnergyTerm("c1*(root-1e-12)+c2*rho*root+c3*rho^2;root=sqrt(max(rho,1e-24))", mm.CustomGBForce.SingleParticle)
        force.addEnergyTerm("phi(r)", mm.CustomGBForce.ParticlePairNoExclusions)
        for atom in range(n_atoms):
            force.addParticle([])
        force.setNonbondedMethod(mm.CustomGBForce.CutoffNonPeriodic)
        force.setCutoffDistance(0.55)            # nm
        system.addForce(force)
        return system

    mo.md("Enter the 15 coefficients above, then run the validation cell below.")
    return (custom_system,)


@app.cell(hide_code=True)
def implementation_check(
    Atoms,
    coefficients,
    custom_system,
    feature_rows,
    mo,
    np,
    openmm_energy_forces,
):
    try:
        custom_system(4)
    except Exception as _error:
        mo.stop(True,mo.md("**Complete the coefficient array above to run these checks.** " + str(_error)))
    check_cluster = Atoms("Cu4", positions=[[0, 0, 0], [2.56, 0, 0], [1.28, 2.2, 0], [1.28, 0.7, 2.1]])
    check_energy_row, check_force_rows = feature_rows(check_cluster)
    check_analytic_energy = float(check_energy_row @ coefficients)
    check_analytic_forces = check_force_rows @ coefficients
    check_openmm_energy, check_openmm_forces = openmm_energy_forces(custom_system(4), check_cluster.positions)
    check_plus, check_minus = check_cluster.copy(), check_cluster.copy()
    check_h = 1e-4
    check_plus.positions[0, 0] += check_h
    check_minus.positions[0, 0] -= check_h
    check_fd_force = -(feature_rows(check_plus)[0] @ coefficients - feature_rows(check_minus)[0] @ coefficients) / (2 * check_h)
    check_force_difference = float(np.max(np.abs(check_openmm_forces - check_analytic_forces)))
    check_isolated_energy, check_isolated_forces = openmm_energy_forces(custom_system(1), np.zeros((1, 3)))
    check_dimer_energy_errors = []
    check_dimer_force_errors = []
    for _distance in (2.2, 3.0, 4.0, 4.8, 5.2, 5.4, 5.49, 6.0):
        _atoms = Atoms("Cu2", positions=[[0, 0, 0], [_distance, 0, 0]])
        _energy_features, _force_features = feature_rows(_atoms)
        _energy, _forces = openmm_energy_forces(custom_system(2), _atoms.positions)
        check_dimer_energy_errors.append(abs(_energy - _energy_features @ coefficients))
        check_dimer_force_errors.append(float(np.max(np.abs(_forces - _force_features @ coefficients))))
    check_separated_energy, check_separated_forces = openmm_energy_forces(custom_system(2), np.array([[0., 0., 0.], [8., 0., 0.]]))
    mo.md(f"""
    ### Check the copied function and the isolated-atom limit

    - Direct fitted 4-atom cluster energy: **{check_analytic_energy:.7f} eV**.
    - OpenMM energy from the copied function: **{check_openmm_energy:.7f} eV**.
    - Largest cluster force-component difference: **{check_force_difference:.2g} eV/Å**.
    - Atom 1 x-force from the direct model: **{check_analytic_forces[0,0]:.7f} eV/Å**.
    - The same force from a finite difference: **{check_fd_force:.7f} eV/Å**.
    - Isolated-atom interaction energy: **{check_isolated_energy:.2g} eV**.
    - Interaction energy of 2 atoms separated by 8 Å: **{check_separated_energy:.2g} eV**.
    - Largest dimer energy difference over 2.2–6.0 Å: **{max(check_dimer_energy_errors):.2g} eV**.
    - Largest dimer force-component difference: **{max(check_dimer_force_errors):.2g} eV/Å**.

    The remaining small interpolation differences come from the pair table.
    """)
    return


@app.cell(hide_code=True)
def paste_instructions(mo):
    mo.md(r"""
    ## 10 · Use your fitted potential in notebook 1

    1. Keep your Mishin results for comparison. Open your own copy of **notebook 1**.
    2. Find **Potential · Mishin or your fitted model**, then **Paste your custom potential here**.
    3. Replace that cell's placeholder `custom_system` with the function you completed above. Run the cell.
    4. Choose **Custom** in the potential dropdown. Its indicator checks whether the function can build
       an OpenMM system with finite energy and forces. If the indicator reports an error, fix the function
       before starting a simulation.
    5. Start with a short fixed-temperature run. Compare the fitted and Mishin models with the same
       particle, seed, time step, and ramp settings.

    Locate the heating transition first, then use a short slow ramp with a solid branch below the rise
    and a liquid branch above it. Extend cooling far enough when investigating freezing. Compare the melting curves and atomic structures from the fitted and Mishin models.
    The supplied held-out structures cover distorted crystals and a surface; the melting runs test the
    model as the particle becomes liquid.
    """)
    return


@app.cell(hide_code=True)
def atomic_reference(atomic_reference_path, json):
    dft_atom_record = json.loads(atomic_reference_path.read_text())
    dft_atom_energy = float(dft_atom_record["isolated_atom_energy_eV"])
    return (dft_atom_energy,)


@app.cell(hide_code=True)
def atom_selection_state(mo):
    get_selected_atom, set_selected_atom = mo.state(1)

    def remember_atom(points):
        if points:
            point = points[-1]
            set_selected_atom(int(point.get("pointIndex", point.get("pointNumber", 1))))

    return get_selected_atom, remember_atom


if __name__ == "__main__":
    app.run()

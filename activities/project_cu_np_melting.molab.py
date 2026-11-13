# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ase==3.29.0",
#     "cuda-toolkit[crt,cudart,cufft,cupti,nvcc,nvjitlink,nvrtc,nvvm]==13.0.2",
#     "marimo==0.25.2",
#     "matplotlib==3.11.1",
#     "numpy==2.4.6",
#     "openmm[cuda13]==8.6.1",
#     "scipy==1.18.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Melting and freezing of Cu nanoparticles

    How do particle size, shape, and heating rate change the melting temperature of copper?
    Use the controls below to run two molecular-dynamics experiments.

    1. Choose a copper particle.
    2. Hold it at a fixed temperature and examine the energy curves.
    3. Heat and cool it, then compare the atomic structures.
    4. Download your results for the analysis notebook.

    Each experiment starts only when you press its **Run** button. Download results you want to keep:
    repeating identical settings replaces the corresponding files in `runs/`, in the **left folder panel**.
    """)
    return


@app.cell(hide_code=True)
def runtime_status(
    PLATFORM_NAME,
    PLATFORM_PROPERTIES,
    mo,
    platform_failures,
    potential_choice,
):
    mo.vstack([
        mo.md(f"**Potential:** `{potential_choice.value}` · **Compute:** {PLATFORM_NAME} · **Time step:** 2 fs"),
        mo.md("The molab GPU is ready." if PLATFORM_NAME == "CUDA" else
              "A GPU is not active. Select a GPU machine in molab for larger particles or slower cycles. "
              "The 2 nm first run can use the CPU fallback."),
        mo.accordion({"Compute details": mo.md("\n\n".join(platform_failures) or str(PLATFORM_PROPERTIES))}),
    ])
    return


@app.cell(hide_code=True)
def potential_intro(mo):
    mo.md("""
    ## Potential · EAM (Mishin et al.) or your fitted model

    Use **EAM (Mishin et al.)** for the main study. For the bonus, enter the 15 fitted coefficients in the
    custom_system skeleton in [notebook 3](https://molab.marimo.io/notebooks/nb_6yiNfKPqBZQSU8LNjCPHco).
    Copy your completed function into the cell below, run it, and choose **Custom**.

    ### Paste your custom potential here

    Replace custom_system = None with your completed function. The readiness message checks its
    energy and forces before MD starts. Custom result filenames include the suffix _custom.
    """)
    return


@app.cell
def custom_potential():
    ###############################################################################
    # CUSTOM MD SYSTEM — BONUS ONLY
    # Implement your custom MD system here, for example using Notebook 3.
    # Follow the bonus instructions: enter your 15 fitted coefficients in the
    # Notebook 3 function, then replace the line below with that completed function.
    # For the main project, leave this cell unchanged and select
    # "EAM (Mishin et al.)" in the Potential dropdown.
    ###############################################################################
    custom_system = None
    return (custom_system,)


@app.cell(hide_code=True)
def potential_controls(mo):
    potential_choice = mo.ui.dropdown(["EAM (Mishin et al.)", "Custom"],
                                      value="EAM (Mishin et al.)", label="Potential")
    potential_choice
    return (potential_choice,)


@app.cell(hide_code=True)
def potential_validation(
    PLATFORM,
    PLATFORM_PROPERTIES,
    custom_system,
    mishin_system,
    mm,
    mo,
    np,
    potential_choice,
    unit,
):
    def check_system_builder(builder):
        if not callable(builder):
            return False, "No custom function yet. Paste the function from the bonus notebook into the marked cell."
        try:
            probe_system = builder(4)
            if not isinstance(probe_system, mm.System):
                return False, "custom_system must return an OpenMM System."
            if probe_system.getNumParticles() != 4:
                return False, "custom_system(4) must create 4 particles."
            if probe_system.getNumForces() == 0:
                return False, "The returned System has no forces."
            probe_integrator = mm.VerletIntegrator(0.002)
            probe_context = mm.Context(probe_system, probe_integrator, PLATFORM, PLATFORM_PROPERTIES)
            probe_context.setPositions(np.array([[0, 0, 0], [0.256, 0, 0], [0.128, 0.22, 0], [0.128, 0.07, 0.21]]))
            probe_state = probe_context.getState(getEnergy=True, getForces=True)
            probe_energy = probe_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            probe_forces = probe_state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            finite = np.isfinite(probe_energy) and np.isfinite(probe_forces).all()
            del probe_context
            if not finite:
                return False, "The startup check produced a non-finite energy or force."
            return True, "Startup check passed: the System has finite energy and forces on a 4-atom cluster."
        except Exception as error:
            # A pasted student function can fail; report its error and block MD rather than switch models.
            return False, f"{type(error).__name__}: {error}"

    selected_builder = mishin_system if potential_choice.value == "EAM (Mishin et al.)" else custom_system
    potential_ready, potential_message = check_system_builder(selected_builder)
    eam_system = selected_builder if potential_ready else None
    mo.callout(
        mo.md(f"**{potential_choice.value} · {'ready' if potential_ready else 'not ready'}** — {potential_message}"),
        kind="success" if potential_ready else "danger",
    )
    return eam_system, potential_ready


@app.cell(hide_code=True)
def particle_md(mo):
    mo.md(r"""
    ## 1 · Choose a particle

    Start with a **2 nm sphere**. The diameter choices give approximately equal atom counts across shapes,
    so use the displayed **equal-volume diameter** when comparing shapes in your report.

    - **Sphere:** a ball cut from an fcc crystal, with steps and corners on its surface.
    - **Cuboctahedron, truncated octahedron, octahedron:** fcc crystals with flat faces.
    - **Icosahedron, Marks decahedron:** 20 or 5 crystal regions joined along twin planes.

    The **random seed** sets the random thermal motion. Change only the seed for an independent repeat.
    """)
    return


@app.cell(hide_code=True)
def particle_controls(mo):
    SHAPES = {
        "Sphere": "sphere",
        "Cuboctahedron": "cuboctahedron",
        "Truncated octahedron": "truncoct",
        "Octahedron": "octahedron",
        "Icosahedron": "icosahedron",
        "Marks decahedron": "decahedron",
    }
    shape_choice = mo.ui.dropdown(SHAPES, value="Sphere", label="Particle shape")
    size_choice = mo.ui.dropdown({"2 nm": 2, "3 nm": 3, "4 nm": 4, "5 nm": 5, "6 nm": 6, "8 nm": 8},
                                 value="2 nm", label="Diameter")
    seed_choice = mo.ui.number(start=1, stop=999, value=1, label="Random seed")
    mo.hstack([shape_choice, size_choice, seed_choice], justify="start", gap=2)
    return seed_choice, shape_choice, size_choice


@app.cell(hide_code=True)
def particle(CU_A0, build_particle, mo, np, plt, shape_choice, size_choice):
    particle = build_particle(shape_choice.value, size_choice.value)
    _d_equiv = (6 * len(particle) * CU_A0**3 / 4 / np.pi) ** (1 / 3) / 10

    _azimuth, _elevation = np.deg2rad([35, 20])
    _rz = np.array([[np.cos(_azimuth), -np.sin(_azimuth), 0],
                    [np.sin(_azimuth), np.cos(_azimuth), 0], [0, 0, 1]])
    _rx = np.array([[1, 0, 0], [0, np.cos(_elevation), -np.sin(_elevation)],
                    [0, np.sin(_elevation), np.cos(_elevation)]])
    _view = particle.positions @ _rz.T @ _rx.T
    _order = np.argsort(_view[:, 2])
    _span = np.ptp(_view[:, :2], axis=0).max()
    _marker_area = (1.7 * 4.2 * 72 / (_span + 3)) ** 2
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    _ax.scatter(_view[_order, 0], _view[_order, 1], s=_marker_area,
                c="#b87333", edgecolors="#73401f", linewidths=0.15, rasterized=True)
    _ax.set_aspect("equal")
    _ax.set_axis_off()
    _bar_A = 5.0 if size_choice.value <= 3 else 10.0
    _x0, _y0 = _view[:,0].min(), _view[:,1].min()-3.0
    _ax.plot([_x0,_x0+_bar_A],[_y0,_y0],color="black",lw=3,solid_capstyle="butt")
    _ax.text(_x0+_bar_A/2,_y0-0.9,f"{_bar_A/10:g} nm",ha="center",va="top",fontsize=10)
    _ax.set_ylim(_y0-3.0,_view[:,1].max()+3.0)
    _fig.tight_layout()
    _preview = _fig

    mo.hstack([
        _preview,
        mo.md(f"""
    **{shape_choice.selected_key}**

    - {len(particle)} atoms
    - equal-volume diameter {_d_equiv:.2f} nm


    """),
    ], justify="start", align="center", wrap=True)
    return (particle,)


@app.cell(hide_code=True)
def run_identity(potential_choice, seed_choice, shape_choice, size_choice):
    seed = int(seed_choice.value)
    tag = f"{shape_choice.value}_{size_choice.value}nm_seed{seed}"
    if potential_choice.value == "Custom":
        tag += "_custom"
    return seed, tag


@app.cell(hide_code=True)
def exp1_md(mo):
    mo.md(r"""
    ## 2 · Energy at a fixed temperature

    The particle starts with its atoms at rest. A thermostat supplies random kicks and friction to
    bring its kinetic energy to the chosen temperature. The atoms also move away from their initial
    crystal positions, which changes the potential energy.

    Choose **900 K** and **20 ps**, then press **Run at fixed temperature**.
    The notebook records the energies every **0.2 ps** and saves **one CSV file**.
    """)
    return


@app.cell(hide_code=True)
def exp1_controls(mo):
    stability_T = mo.ui.slider(300, 1300, step=50, value=900,
                              label="Temperature (K)", show_value=True)
    stability_ps = mo.ui.dropdown({"20 ps": 20, "50 ps": 50, "100 ps": 100},
                                 value="20 ps", label="Duration")
    run_stability = mo.ui.run_button(label="Run at fixed temperature")
    mo.hstack([stability_T, stability_ps, run_stability], justify="start", gap=2)
    return run_stability, stability_T, stability_ps


@app.cell(hide_code=True)
def exp1_run(
    download_buttons,
    experiment,
    mo,
    particle,
    potential_ready,
    run_md,
    run_stability,
    seed,
    stability_T,
    stability_ps,
    tag,
):
    mo.stop(not potential_ready,mo.md("Select a ready potential before running MD."))
    stability_name = f"stability_{tag}_T{stability_T.value}_{stability_ps.value}ps"
    stability = experiment(
        stability_name,
        lambda: run_md(particle, [("hold", float(stability_T.value), 0.2)] * (5 * stability_ps.value), seed,
                       title=f"{stability_ps.value} ps at {stability_T.value} K", start_at_rest=True),
        run_stability.value,
    )
    (mo.md("Choose a temperature and a length, then press **Run at fixed temperature**.")
     if stability is None else download_buttons(stability_name))
    return (stability,)


@app.cell(hide_code=True)
def exp1_plot(
    KB_EV,
    mo,
    particle,
    plt,
    seed,
    shape_choice,
    stability,
    stability_T,
):
    mo.stop(stability is None)
    _fig, (_ax_p, _ax_k) = plt.subplots(1, 2, figsize=(10, 3.4))
    _ax_p.plot(stability["time_ps"], stability["PE_eV_per_atom"], lw=1, color="tab:red")
    _ax_p.set(xlabel="Time (ps)", ylabel="Potential energy (eV/atom)")
    _ax_k.plot(stability["time_ps"], stability["KE_eV_per_atom"], lw=1, color="tab:blue")
    _ax_k.axhline(1.5 * KB_EV * stability_T.value, color="k", ls="--", lw=1, label=r"$\frac{3}{2}k_\mathrm{B}T$")
    _ax_k.set(xlabel="Time (ps)", ylabel="Kinetic energy (eV/atom)")
    _ax_k.legend(frameon=False)
    _fig.suptitle(f"{shape_choice.selected_key}, {len(particle)} atoms, {stability_T.value} K, seed {seed}", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def exp1_questions(mo):
    mo.md(r"""
    ### What do the energy curves show?

    - After how long does each energy fluctuate about a steady mean?
    - Does the mean kinetic energy per atom approach $\tfrac32 k_\mathrm{B}T$, the dashed line?
      At 900 K this is **0.1163 eV/atom**.
    - Repeat with a different seed. How much does the estimated mean change?

    Download the CSV before moving on. It contains time, set and measured temperatures, and both energies per atom.
    """)
    return


@app.cell(hide_code=True)
def exp2_md(mo):
    mo.md(r"""
    ## 3 · Heating and cooling

    For a first cycle, use **400 K to 1600 K** at **10 K/ps**. The particle passes through four stages:

    1. **Equilibrate:** 100 ps at your chosen start temperature, with a separate energy curve.
    2. **Heat:** increase the thermostat temperature by 1 K at each step until it reaches your chosen maximum.
    3. **Hold:** 20 ps at the maximum temperature.
    4. **Cool:** return to the start temperature at the same rate.

    The cycle CSV and XYZ begin after equilibration, with time reset to zero. The initial XYZ frame
    is the equilibrated particle. Equilibration energies are saved in a separate CSV.


    **Hint**: running very slow ramp-rate (below $1 K/ps$ from 400 K to 1600 K could be **VERY SLOW**!) If you already have a rough estimation of the melting point, make the start and maximum temperature close to the desired value!
    """)
    return


@app.cell(hide_code=True)
def cycle_controls(mo):
    RATES = {"10 K/ps": 10.0, "5 K/ps": 5.0, "2 K/ps": 2.0, "1 K/ps": 1.0,
             "0.5 K/ps": 0.5, "0.2 K/ps": 0.2, "0.1 K/ps": 0.1}
    cycle_rate = mo.ui.dropdown(RATES, value="10 K/ps", label="Heating and cooling rate")
    snapshot_spacing = mo.ui.dropdown({"50 K": 50, "100 K": 100, "200 K": 200},
                                     value="100 K", label="Save a structure every")
    run_cycle = mo.ui.run_button(label="Run heating and cooling")
    cycle_start = mo.ui.number(start=100, stop=2500, step=50, value=400, label="Start / cooling end (K)")
    cycle_maximum = mo.ui.number(start=100, stop=2500, step=50, value=1600, label="Maximum temperature (K)")
    mo.vstack([
        mo.hstack([cycle_start, cycle_maximum], justify="start", gap=2, wrap=True),
        cycle_rate,
        snapshot_spacing,
        run_cycle,
    ])
    return


@app.cell(hide_code=True)
def exp2_run(
    RUN_DIR,
    cycle_maximum,
    cycle_rate,
    cycle_start,
    download_buttons,
    experiment,
    load_csv,
    mo,
    particle,
    potential_ready,
    run_cycle,
    run_md,
    seed,
    snapshot_spacing,
    tag,
):
    mo.stop(not potential_ready, mo.md("Select a ready potential before running MD."))
    _low, _high = int(cycle_start.value), int(cycle_maximum.value)
    mo.stop(_high <= _low, mo.md("**Choose a maximum temperature above the start temperature.**"))
    _equilibration_ps = 100.0
    _schedule = cycle_schedule(cycle_rate.value, t_low=_low, t_high=_high)
    _cycle_ps = sum(piece[2] for piece in _schedule)
    _spacing = snapshot_spacing.value
    _snapshots = 1 + sum(
        i == len(_schedule) - 1 or _schedule[i + 1][0] != stage
        or (stage in ("heat", "cool") and abs(temp / _spacing - round(temp / _spacing)) < 1e-8)
        for i, (stage, temp, duration) in enumerate(_schedule)
    )
    cycle_name = f"cycle_{tag}_{cycle_rate.value:g}Kps_{_low}-{_high}K_snap{_spacing}K_eq100ps"
    cycle = experiment(
        cycle_name,
        lambda: run_md(particle, _schedule, seed,
                       title=f"{_low} → {_high} → {_low} K at {cycle_rate.value:g} K/ps",
                       snapshot_kelvin=_spacing, equilibration_ps=_equilibration_ps,
                       equilibration_temperature=_low),
        run_cycle.value,
    )
    _equilibration_path = RUN_DIR / f"{cycle_name}_equilibration.csv"
    cycle_equilibration = load_csv(_equilibration_path) if _equilibration_path.is_file() else None
    mo.vstack([
        mo.md(f"- Temperature: **{_low} → {_high} → {_low} K**.\n"
              f"- Heating and cooling rate: **{cycle_rate.value:g} K/ps**.\n"
              f"- Initial equilibration: **100 ps**, recorded separately; hot hold: **20 ps**.\n"
              f"- Cycle time: **{_cycle_ps:g} ps** ({_cycle_ps / 1000:.3f} ns), after equilibration.\n"
              f"- Total simulated time including equilibration: **{_cycle_ps + _equilibration_ps:g} ps**.\n"
              f"- Cycle output: **{len(_schedule)} energy records** and **{_snapshots} snapshots**, excluding equilibration."),
        mo.md("Press **Run heating and cooling** to start.") if cycle is None else download_buttons(cycle_name),
        mo.md("The separate equilibration curve will appear below.") if cycle_equilibration is None else
        mo.download(data=lambda p=_equilibration_path: p.read_bytes(),
                    filename=_equilibration_path.name, mimetype="text/csv",
                    label="Download equilibration CSV (separate)"),
    ])

    return cycle, cycle_equilibration, cycle_name


@app.cell(hide_code=True)
def cycle_equilibration_plot(KB_EV, cycle_equilibration, mo, plt):
    _heading = mo.md("### Initial 100 ps equilibration · separate from the cycle")
    mo.stop(cycle_equilibration is None,
            mo.vstack([_heading, mo.md("Run a cycle to display its initial equilibration energies.")]))
    _equilibration_figure, (_pe_axis, _ke_axis) = plt.subplots(1, 2, figsize=(10, 3.4))
    _pe_axis.plot(cycle_equilibration["time_ps"], cycle_equilibration["PE_eV_per_atom"],
                  color="tab:red", lw=1)
    _pe_axis.set(xlabel="Equilibration time (ps)", ylabel="Potential energy (eV/atom)")
    _ke_axis.plot(cycle_equilibration["time_ps"], cycle_equilibration["KE_eV_per_atom"],
                  color="tab:blue", lw=1)
    _ke_axis.axhline(1.5 * KB_EV * cycle_equilibration["T_set_K"][0],
                     color="black", ls="--", lw=1, label="Mean kinetic energy at target T")
    _ke_axis.set(xlabel="Equilibration time (ps)", ylabel="Kinetic energy (eV/atom)")
    _ke_axis.legend(frameon=False)
    _equilibration_figure.tight_layout()
    mo.vstack([_heading, _equilibration_figure])

    return


@app.cell(hide_code=True)
def exp2_plot(cycle, mo, particle, plt, potential_choice, shape_choice):
    _heading = mo.md("### Current heating / cooling cycle in MD")
    mo.stop(cycle is None,mo.vstack([_heading,mo.md("Run heating and cooling to show its potential-energy curve.")]))
    cycle_energy_figure,_ax = plt.subplots(figsize=(7,4))
    for _stage,_colour,_label in (("heat","tab:red","Heating"),("cool","tab:blue","Cooling")):
        _mask = cycle["stage"] == _stage
        if _mask.any():
            _ax.plot(cycle["T_set_K"][_mask],cycle["PE_eV_per_atom"][_mask],color=_colour,lw=1,label=_label)
    _ax.set(xlabel="Thermostat temperature (K)",ylabel="Potential energy (eV/atom)")
    _ax.set_title(f"{shape_choice.selected_key}, {len(particle)} atoms · {potential_choice.value}",fontsize=10)
    _ax.legend(frameon=False)
    cycle_energy_figure.tight_layout()
    mo.vstack([_heading,cycle_energy_figure])
    return (cycle_energy_figure,)


@app.cell(hide_code=True)
def exp2_questions(mo):
    mo.md(r"""
    ### Where do melting and freezing occur?

    - Find the temperature range where the energy rises most steeply on heating.
    - Compare the cooling curve with the heating curve. Does freezing occur at the same temperature?
    - Use the snapshots below to compare the initial crystal, the hot particle, and the cooled particle.
      What structural change accompanies the energy change?

    Download **both files** for the analysis notebook. Use slower cycles and repeated seeds to
    investigate the transition temperatures for your report.
    """)
    return


@app.cell(hide_code=True)
def viewer_md(mo):
    mo.md(r"""
    ## 4 · Compare atomic structures

    Choose a cycle file and move the frame slider. Each picture shows a **4 Å-thick central slice**
    through the particle. Compare the regular lattice with the atomic arrangement after heating and cooling.
    """)
    return


@app.cell(hide_code=True)
def viewer_controls(RUN_DIR, cycle, cycle_name, mo):
    # Refresh the list when a new cycle finishes. Changing the particle alone saves no files.
    _ = cycle
    _xyz_files = sorted(p.name for p in RUN_DIR.glob("cycle_*.xyz"))
    _preferred = f"{cycle_name}.xyz"
    _selected = _preferred if _preferred in _xyz_files else (_xyz_files[-1] if _xyz_files else None)
    viewer_file = mo.ui.dropdown(_xyz_files, value=_selected, label="Cycle structure file")
    viewer_file
    return (viewer_file,)


@app.cell(hide_code=True)
def viewer_frames(RUN_DIR, mo, read_structures, viewer_file):
    mo.stop(viewer_file.value is None, mo.md("Run an experiment first."))
    saved_frames = read_structures(RUN_DIR / viewer_file.value, index=":")
    saved_frame = mo.ui.slider(0, max(1, len(saved_frames) - 1), value=0,
                               disabled=len(saved_frames) == 1, label="Frame", show_value=True)
    mo.vstack([mo.md(f"`{viewer_file.value}`: {len(saved_frames)} frames, {len(saved_frames[0])} atoms"), saved_frame])
    return saved_frame, saved_frames


@app.cell(hide_code=True)
def viewer(np, plot_atoms, plt, saved_frame, saved_frames):
    _frame = saved_frames[saved_frame.value]
    _xyz = _frame.positions - _frame.positions.mean(axis=0)
    _slice = np.abs(_xyz[:, 2]) < 2.0
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    _atoms = _frame[_slice]
    _atoms.positions = _xyz[_slice]
    plot_atoms(_atoms, _ax, rotation="0x,0y,0z", show_unit_cell=0, radii=0.55)
    _ax.set_title(f"{_frame.info.get('stage', '')} · {_frame.info.get('T_set_K', '')} K"
                      f" · {_frame.info.get('time_ps', '')} ps")
    _ax.set_aspect("equal")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def local_xyz_viewing(mo):
    mo.md("""
    ### Optional 3D visualization with OVITO

    Use [OVITO Basic](https://www.ovito.org/) to rotate the particle and play the downloaded XYZ snapshots.
    See **Optional · View the XYZ frames locally with OVITO** in the project tutorial for instructions.
    """)
    return


@app.cell(hide_code=True)
def imports():
    import csv
    import time
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import openmm as mm
    import openmm.unit as unit
    from pathlib import Path
    from ase import Atoms
    from ase.cluster import Decahedron, Icosahedron, Octahedron
    from ase.io import read as read_structures, write as write_structures
    from ase.visualize.plot import plot_atoms
    from scipy.spatial import cKDTree

    NOTEBOOK_DIR = Path(mo.notebook_dir() or Path.cwd())
    RUN_DIR = NOTEBOOK_DIR / "runs"
    RUN_DIR.mkdir(exist_ok=True)
    POTENTIAL = NOTEBOOK_DIR / "Cu_mishin1.eam.alloy"
    if not POTENTIAL.is_file():
        raise FileNotFoundError("The provided Cu_mishin1.eam.alloy file is missing from this molab copy.")
    return (
        Atoms,
        Decahedron,
        Icosahedron,
        Octahedron,
        POTENTIAL,
        RUN_DIR,
        csv,
        mm,
        mo,
        np,
        plot_atoms,
        plt,
        read_structures,
        time,
        unit,
        write_structures,
    )


@app.cell(hide_code=True)
def potential(POTENTIAL, mm, np):
    EV_TO_KJMOL = 96.48533212  # kJ/mol per eV
    KB_EV = 8.617333262e-5     # Boltzmann constant, eV/K
    CU_MASS = 63.546           # g/mol
    CU_A0 = 3.615              # Angstrom, fcc lattice constant of the Mishin potential


    def read_setfl(path):
        """Read a one-element DYNAMO setfl (.eam.alloy) file: F(rho), f(r), phi(r)."""
        with open(path) as fh:
            lines = fh.readlines()
        nrho, drho, nr, dr, rcut = lines[4].split()
        nrho, nr, drho, dr, rcut = int(nrho), int(nr), float(drho), float(dr), float(rcut)
        values = np.array(" ".join(lines[6:]).split(), dtype=float)
        embed = values[:nrho]
        density = values[nrho:nrho + nr]
        r_phi = values[nrho + nr:nrho + 2 * nr]   # the file stores r * phi(r)
        r = np.arange(nr) * dr
        phi = np.empty_like(r_phi)
        phi[1:] = r_phi[1:] / r[1:]
        phi[0] = phi[1]
        return dict(embed=embed, density=density, phi=phi,
                    rho_max=(nrho - 1) * drho, r_max=(nr - 1) * dr, cutoff=rcut)


    eam = read_setfl(POTENTIAL)


    def mishin_system(n_atoms):
        """OpenMM System of a free Cu cluster (no periodic box) with the EAM energy
        E = sum_i F(rho_i) + sum_{i<j} phi(r_ij),  rho_i = sum_j f(r_ij)."""
        system = mm.System()
        for _ in range(n_atoms):
            system.addParticle(CU_MASS)
        force = mm.CustomGBForce()
        force.addTabulatedFunction("fden", mm.Continuous1DFunction(eam["density"], 0.0, eam["r_max"] / 10))
        force.addTabulatedFunction("phi", mm.Continuous1DFunction(eam["phi"] * EV_TO_KJMOL, 0.0, eam["r_max"] / 10))
        force.addTabulatedFunction("Femb", mm.Continuous1DFunction(eam["embed"] * EV_TO_KJMOL, 0.0, eam["rho_max"]))
        force.addComputedValue("rho", "fden(r)", mm.CustomGBForce.ParticlePairNoExclusions)
        force.addEnergyTerm("Femb(rho)", mm.CustomGBForce.SingleParticle)
        force.addEnergyTerm("phi(r)", mm.CustomGBForce.ParticlePairNoExclusions)
        for _ in range(n_atoms):
            force.addParticle([])
        force.setNonbondedMethod(mm.CustomGBForce.CutoffNonPeriodic)
        force.setCutoffDistance(eam["cutoff"] / 10)
        system.addForce(force)
        return system


    def pick_platform():
        """Use the molab GPU when available. Keep a 4-thread CPU fallback for a short run."""
        failures = []
        for name, properties in (("CUDA", {"Precision": "mixed"}), ("CPU", {"Threads": "4"})):
            try:
                platform = mm.Platform.getPlatformByName(name)
                probe = mm.Context(mishin_system(2), mm.VerletIntegrator(0.001), platform, properties)
                del probe
                return name, platform, properties, failures
            except Exception as error:
                failures.append(f"{name}: {error}")
        raise RuntimeError("OpenMM could not start. " + " | ".join(failures))


    PLATFORM_NAME, PLATFORM, PLATFORM_PROPERTIES, platform_failures = pick_platform()
    return (
        CU_A0,
        EV_TO_KJMOL,
        KB_EV,
        PLATFORM,
        PLATFORM_NAME,
        PLATFORM_PROPERTIES,
        mishin_system,
        platform_failures,
    )


@app.cell(hide_code=True)
def md_functions(
    Atoms,
    EV_TO_KJMOL,
    KB_EV,
    PLATFORM,
    PLATFORM_PROPERTIES,
    RUN_DIR,
    csv,
    eam_system,
    mm,
    mo,
    np,
    time,
    unit,
    write_structures,
):
    COLUMNS = ["time_ps", "stage", "T_set_K", "T_K", "PE_eV_per_atom", "KE_eV_per_atom"]


    def run_md(atoms, schedule, seed, title, start_at_rest=False, snapshot_kelvin=None,
               dt_fs=2.0, friction_per_ps=2.0, equilibration_ps=0.0, equilibration_temperature=None):
        """Return production data, production frames, and a separate equilibration table.

        Equilibration and production share one Context and continuous positions/velocities.
        Production time starts at zero after equilibration. No equilibration frame is exported.
        """
        started = time.perf_counter()
        n = len(atoms)
        start_temperature = schedule[0][1] if equilibration_temperature is None else equilibration_temperature
        integrator = mm.LangevinMiddleIntegrator(
            start_temperature * unit.kelvin, friction_per_ps / unit.picosecond,
            dt_fs * unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(seed)
        context = mm.Context(eam_system(n), integrator, PLATFORM, PLATFORM_PROPERTIES)

        def frame(positions_A, elapsed_ps, stage, t_set):
            return Atoms(f"Cu{n}", positions=positions_A - positions_A.mean(axis=0),
                         info={"time_ps": round(elapsed_ps, 3), "stage": stage, "T_set_K": round(t_set, 1)})

        def advance(steps):
            # Keep long, slow-ramp intervals interruptible without changing the time step.
            chunk = max(1, round(1000 / dt_fs))  # at most 1 ps per OpenMM call
            remaining = steps
            while remaining:
                count = min(chunk, remaining)
                integrator.step(count)
                remaining -= count
            return steps * dt_fs / 1000

        def record(table, elapsed_ps, stage, t_set, state):
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / EV_TO_KJMOL / n
            ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / EV_TO_KJMOL / n
            for column, value in zip(COLUMNS, (elapsed_ps, stage, t_set, 2 * ke / (3 * KB_EV), pe, ke)):
                table[column].append(value)

        try:
            context.setPositions(atoms.positions / 10)  # Angstrom -> nm
            if not start_at_rest:
                context.setVelocitiesToTemperature(start_temperature * unit.kelvin, seed)

            equilibration = None
            if equilibration_ps > 0:
                equilibration = {column: [] for column in COLUMNS}
                total_steps = round(equilibration_ps * 1000 / dt_fs)
                log_steps = max(1, round(0.2 * 1000 / dt_fs))
                equilibration_time = 0.0
                for first in mo.status.progress_bar(
                    range(0, total_steps, log_steps),
                    title=f"Equilibrating at {start_temperature:g} K for {equilibration_ps:g} ps",
                    remove_on_exit=True,
                ):
                    equilibration_time += advance(min(log_steps, total_steps - first))
                    state = context.getState(getEnergy=True)
                    record(equilibration, equilibration_time, "equilibrate", start_temperature, state)

            # The initial trajectory frame is the equilibrated configuration, at production t=0.
            table = {column: [] for column in COLUMNS}
            frames = []
            if snapshot_kelvin:
                initial_state = context.getState(getPositions=True)
                frames.append(frame(initial_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                                    0.0, "initial", start_temperature))
            time_ps = 0.0
            pieces = list(enumerate(schedule))
            for piece_index, (stage, t_set, length_ps) in mo.status.progress_bar(
                pieces, title=title, remove_on_exit=True,
            ):
                integrator.setTemperature(t_set * unit.kelvin)
                time_ps += advance(max(1, round(length_ps * 1000 / dt_fs)))
                stage_end = piece_index == len(schedule) - 1 or schedule[piece_index + 1][0] != stage
                at_interval = (stage in ("heat", "cool")
                               and abs(t_set / (snapshot_kelvin or 1) - round(t_set / (snapshot_kelvin or 1))) < 1e-8)
                want_frame = bool(snapshot_kelvin) and (stage_end or at_interval)
                state = context.getState(getEnergy=True, getPositions=want_frame)
                record(table, time_ps, stage, t_set, state)
                if want_frame:
                    frames.append(frame(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                                        time_ps, stage, t_set))
        finally:
            del context
            del integrator

        elapsed = time.perf_counter() - started
        print(f"{title}: {n} atoms, {equilibration_ps:g} ps equilibration + {time_ps:.3f} ps cycle, "
              f"{elapsed:.2f} s, {len(table['time_ps'])} recorded rows, {len(frames)} snapshots")
        return table, frames, equilibration


    def save_csv(path, table):
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(COLUMNS)
            for row in zip(*(table[c] for c in COLUMNS)):
                writer.writerow([row[0] if isinstance(row[0], str) else f"{row[0]:.3f}", row[1],
                                 *(f"{v:.6f}" for v in row[2:])])


    def load_csv(path):
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        return {c: np.array([r[c] for r in rows]) if c == "stage" else np.array([float(r[c]) for r in rows])
                for c in COLUMNS}


    def experiment(name, compute, run_pressed):
        """Run an experiment when its button is pressed and save runs/<name>.csv (and .xyz).
        Otherwise load the saved files. Returns None when there is nothing yet."""
        csv_path, xyz_path = RUN_DIR / f"{name}.csv", RUN_DIR / f"{name}.xyz"
        if run_pressed:
            table, frames, equilibration = compute()
            if equilibration is not None:
                save_csv(RUN_DIR / f"{name}_equilibration.csv", equilibration)
            save_csv(csv_path, table)
            if frames:
                write_structures(xyz_path, frames, format="extxyz")
            print(f"Saved {csv_path.name}: {csv_path.stat().st_size} bytes"
                  + (f"; {xyz_path.name}: {xyz_path.stat().st_size} bytes" if frames else ""))
        return load_csv(csv_path) if csv_path.exists() else None


    def download_buttons(name):
        """Download buttons for the files of one experiment, and where to find them in the file panel."""
        buttons = []
        for suffix, mimetype, label in ((".csv", "text/csv", "Download data (.csv)"),
                                        (".xyz", "chemical/x-xyz", "Download structures (.xyz)")):
            path = RUN_DIR / f"{name}{suffix}"
            if path.exists():
                buttons.append(mo.download(data=lambda p=path: p.read_bytes(), filename=path.name,
                                           mimetype=mimetype, label=label))
        return mo.hstack([*buttons, mo.md(f"File panel: `runs/{name}.*`")], justify="start", align="center", gap=1)

    return download_buttons, experiment, load_csv, run_md


@app.cell(hide_code=True)
def particle_builders(Atoms, CU_A0, Decahedron, Icosahedron, Octahedron, np):
    def sphere(diameter_nm, a=CU_A0):
        """fcc atoms inside a sphere centred on an atom."""
        n = int(np.ceil(diameter_nm * 5 / a)) + 1
        basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        grid = np.array(np.meshgrid(*[np.arange(-n, n)] * 3, indexing="ij")).reshape(3, -1).T
        x = (grid[:, None, :] + basis[None]).reshape(-1, 3) * a
        return Atoms(symbols="Cu" + str(int((np.linalg.norm(x, axis=1) <= diameter_nm * 5).sum())),
                     positions=x[np.linalg.norm(x, axis=1) <= diameter_nm * 5])


    # ASE builder settings that give about the same number of atoms as a sphere of each diameter
    BUILDER_SETTINGS = {
        #     octahedron  truncated oct.  cuboctahedron  icosahedron  Marks decahedron
        2: dict(octahedron=(8, 0), truncoct=(9, 3), cuboctahedron=(9, 4), icosahedron=5, decahedron=(5, 2, 1)),
        3: dict(octahedron=(12, 0), truncoct=(13, 5), cuboctahedron=(15, 7), icosahedron=8, decahedron=(6, 3, 2)),
        4: dict(octahedron=(16, 0), truncoct=(17, 5), cuboctahedron=(19, 9), icosahedron=10, decahedron=(9, 4, 2)),
        5: dict(octahedron=(20, 0), truncoct=(21, 7), cuboctahedron=(23, 11), icosahedron=12, decahedron=(11, 6, 2)),
        6: dict(octahedron=(24, 0), truncoct=(25, 8), cuboctahedron=(29, 14), icosahedron=15, decahedron=(15, 9, 1)),
        8: dict(octahedron=(32, 0), truncoct=(34, 12), cuboctahedron=(37, 18), icosahedron=19, decahedron=(19, 11, 2)),
    }


    def build_particle(shape, diameter_nm):
        """Cu particle of the chosen shape with about as many atoms as a sphere of diameter_nm (ASE builders)."""
        settings = BUILDER_SETTINGS[diameter_nm]
        if shape == "sphere":
            atoms = sphere(diameter_nm)
        elif shape in ("octahedron", "truncoct", "cuboctahedron"):
            atoms = Octahedron("Cu", *settings[shape], latticeconstant=CU_A0)
        elif shape == "icosahedron":
            atoms = Icosahedron("Cu", settings[shape], latticeconstant=CU_A0)
        else:
            atoms = Decahedron("Cu", *settings[shape], latticeconstant=CU_A0)
        atoms.positions -= atoms.positions.mean(axis=0)
        atoms.cell = None
        atoms.pbc = False
        return atoms


    return (build_particle,)


@app.function(hide_code=True)
def cycle_schedule(rate, t_low=400.0, t_high=1600.0, hold_ps=20.0):
    """Production schedule; initial equilibration is handled separately in run_md."""
    steps = int(round(t_high - t_low))
    return ([("heat", t_low + k + 1, 1.0 / rate) for k in range(steps)]
            + [("hold", t_high, 1.0)] * int(hold_ps)
            + [("cool", t_high - k - 1, 1.0 / rate) for k in range(steps)])


if __name__ == "__main__":
    app.run()

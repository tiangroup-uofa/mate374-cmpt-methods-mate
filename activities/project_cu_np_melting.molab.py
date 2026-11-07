# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.25.2",
#     "ase==3.29.0",
#     "matplotlib==3.11.2",
#     "numpy==2.5.4",
#     "openmm[cuda12]==8.6.1",
#     "scipy==1.18.1",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Melting and freezing of Cu nanoparticles

    How do particle size, shape, and heating rate affect the temperature at which copper melts?
    In this notebook, you run molecular dynamics of free copper nanoparticles using the Mishin EAM1
    potential (Mishin et al., *Phys. Rev. B* **63**, 224106, 2001).

    1. Choose a particle shape, size, and random seed.
    2. Hold it at a fixed temperature and examine its energy over time.
    3. Heat and cool it, then examine its energy and atomic structures.
    4. Download the results and use the analysis notebook to compare your group's runs.

    Locally, open a terminal in this folder and run `uvx marimo@0.25.2 edit --sandbox project_cu_np_melting.molab.py`.
    The sandbox installs the declared dependencies. OpenMM selects NVIDIA CUDA when available and otherwise
    uses multiple CPU threads. Use the platform selector to compare them. ASE renders the atomic
    structures in the notebook through its matplotlib plotting interface.

    Download each result before changing its settings: running the same settings again replaces files
    with the same names. Local results also remain in the `runs/` folder.
    """)
    return


@app.cell
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

    # Run without the browser, for example:
    #   python notebook.py --shape icosahedron --seed 2 --rate 1 --run cycle
    #   uv run notebook.py --temperature 1100 --length 100 --run stability,cycle
    # Options: --shape --size --seed --temperature --length --rate --snapshots --platform --threads --run.
    # Use --platform CUDA to require the GPU, or --platform CPU --threads 4 for a CPU run.
    ARGS = mo.cli_args()
    IS_SCRIPT = mo.app_meta().mode == "script"


    def script_runs(name):
        """True when the notebook runs as a script and --run lists this experiment."""
        return IS_SCRIPT and name in str(ARGS.get("run", "")).split(",")


    NOTEBOOK_DIR = Path(mo.notebook_dir() or Path.cwd())
    RUN_DIR = NOTEBOOK_DIR / "runs"          # every experiment saves its files here
    RUN_DIR.mkdir(exist_ok=True)
    POTENTIAL = NOTEBOOK_DIR / "Cu_mishin1.eam.alloy"   # shipped with the notebook
    if not POTENTIAL.exists():
        raise FileNotFoundError("Keep Cu_mishin1.eam.alloy beside this notebook. Extract the complete assignment ZIP first.")

    return (
        ARGS,
        Atoms,
        Decahedron,
        Icosahedron,
        Octahedron,
        POTENTIAL,
        RUN_DIR,
        cKDTree,
        csv,
        mm,
        mo,
        np,
        plt,
        plot_atoms,
        read_structures,
        script_runs,
        time,
        unit,
        write_structures,
    )


@app.cell
def platform_controls(ARGS, mo):
    platform_choice = mo.ui.dropdown(["Auto", "CUDA", "CPU"], value=str(ARGS.get("platform", "Auto")), label="OpenMM platform")
    cpu_threads = mo.ui.number(start=1, stop=64, value=int(ARGS.get("threads", 4)), label="CPU threads")
    mo.hstack([platform_choice, cpu_threads], justify="start")
    return cpu_threads, platform_choice


@app.cell
def potential(POTENTIAL, cpu_threads, mm, mo, np, platform_choice):
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


    def eam_system(n_atoms):
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
        """Probe a context, report failures, and never silently replace an explicitly requested platform."""
        names = ["CUDA", "CPU"] if platform_choice.value == "Auto" else [platform_choice.value]
        failures = []
        for name in names:
            properties = {"Precision": "mixed"} if name == "CUDA" else {"Threads": str(int(cpu_threads.value))}
            try:
                platform = mm.Platform.getPlatformByName(name)
                probe = mm.Context(eam_system(2), mm.VerletIntegrator(0.001), platform, properties)
                del probe
                return name, platform, properties, failures
            except Exception as error:
                failures.append(f"{name}: {error}")
        raise RuntimeError("Cannot start the requested OpenMM platform. " + " | ".join(failures))


    PLATFORM_NAME, PLATFORM, PLATFORM_PROPERTIES, platform_failures = pick_platform()
    print(f"OpenMM {mm.__version__}: {PLATFORM_NAME}, properties={PLATFORM_PROPERTIES}")
    mo.md(f"OpenMM {mm.__version__} runs on **{PLATFORM_NAME}**, properties `{PLATFORM_PROPERTIES}`. "
          "Start with a 2 nm particle and a fast cycle before choosing larger or slower runs."
          + ("  \nGPU probe: " + " | ".join(platform_failures) if platform_failures else ""))

    return CU_A0, EV_TO_KJMOL, KB_EV, PLATFORM, PLATFORM_PROPERTIES, eam_system


@app.cell
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
               dt_fs=2.0, friction_per_ps=2.0):
        """Langevin MD of a free Cu particle.

        schedule: list of (stage, set temperature in K, length in ps). The thermostat temperature
        is updated at the start of each piece, and one row of data is recorded at its end.
        Returns the data table and sparse structures at temperature intervals and stage boundaries.
        Snapshot count depends on the temperature range, not the number of integration steps.
        """
        started = time.perf_counter()
        n = len(atoms)
        integrator = mm.LangevinMiddleIntegrator(schedule[0][1] * unit.kelvin, friction_per_ps / unit.picosecond,
                                                 dt_fs * unit.femtoseconds)
        integrator.setRandomNumberSeed(seed)
        context = mm.Context(eam_system(n), integrator, PLATFORM, PLATFORM_PROPERTIES)
        context.setPositions(atoms.positions / 10)          # Angstrom -> nm
        if not start_at_rest:
            context.setVelocitiesToTemperature(schedule[0][1] * unit.kelvin, seed)

        def frame(positions_A, time_ps, stage, t_set):
            return Atoms(f"Cu{n}", positions=positions_A - positions_A.mean(axis=0),
                         info={"time_ps": round(time_ps, 3), "stage": stage, "T_set_K": round(t_set, 1)})

        table = {c: [] for c in COLUMNS}
        frames = [frame(atoms.positions, 0.0, "initial", schedule[0][1])] if snapshot_kelvin else []
        time_ps = 0.0
        pieces = list(enumerate(schedule))
        for piece_index, (stage, t_set, length_ps) in mo.status.progress_bar(pieces, title=title, remove_on_exit=True):
            integrator.setTemperature(t_set * unit.kelvin)
            steps = max(1, round(length_ps * 1000 / dt_fs))
            integrator.step(steps)
            time_ps += steps * dt_fs / 1000
            stage_end = piece_index == len(schedule) - 1 or schedule[piece_index + 1][0] != stage
            at_interval = stage in ("heat", "cool") and abs(t_set / (snapshot_kelvin or 1) - round(t_set / (snapshot_kelvin or 1))) < 1e-8
            want_frame = bool(snapshot_kelvin) and (stage_end or at_interval)
            state = context.getState(getEnergy=True, getPositions=want_frame)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / EV_TO_KJMOL / n
            ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / EV_TO_KJMOL / n
            for c, v in zip(COLUMNS, (time_ps, stage, t_set, 2 * ke / (3 * KB_EV), pe, ke)):
                table[c].append(v)
            if want_frame:
                frames.append(frame(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                                    time_ps, stage, t_set))
        del context
        elapsed = time.perf_counter() - started
        print(f"{title}: {n} atoms, {time_ps:.3f} ps, {elapsed:.2f} s, {len(table['time_ps'])} rows, {len(frames)} snapshots")
        return table, frames


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
            table, frames = compute()
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


    return download_buttons, experiment, run_md


@app.cell(hide_code=True)
def particle_md(mo):
    mo.md(r"""
    ## Step 0 · Choose your particle

    Choose the shape and the diameter. For each diameter, all shapes have about the same number of atoms
    (for example, about 2900 atoms at 4 nm):

    - **Sphere:** a ball cut from the fcc crystal. Its surface has many steps and corners.
    - **Cuboctahedron, truncated octahedron, octahedron:** fcc crystals bounded by flat {111} and {100} facets.
    - **Icosahedron, Marks decahedron:** made of 20 or 5 slightly strained fcc grains joined by twin planes.
      These shapes are common in real Cu and Au nanoparticles of this size.

    The random seed sets the random thermal motion. Two runs with different seeds are independent repeats
    of the same experiment.
    """)

    return


@app.cell
def particle_controls(ARGS, mo):
    SHAPES = {
        "Sphere": "sphere",
        "Cuboctahedron": "cuboctahedron",
        "Truncated octahedron": "truncoct",
        "Octahedron": "octahedron",
        "Icosahedron": "icosahedron",
        "Marks decahedron": "decahedron",
    }
    _shape = str(ARGS.get("shape", "sphere"))
    shape_choice = mo.ui.dropdown(SHAPES, value=next(k for k, v in SHAPES.items() if v == _shape), label="Particle shape")
    SIZES = {"2 nm": 2, "3 nm": 3, "4 nm": 4, "6 nm": 6, "8 nm": 8}
    size_choice = mo.ui.dropdown(SIZES, value=f"{int(ARGS.get('size', 2))} nm", label="Diameter")
    seed_choice = mo.ui.number(start=1, stop=999, value=int(ARGS.get("seed", 1)), label="Random seed")
    mo.hstack([shape_choice, size_choice, seed_choice], justify="start", gap=2)

    return seed_choice, shape_choice, size_choice


@app.cell
def particle(
    Atoms,
    CU_A0,
    Decahedron,
    Icosahedron,
    Octahedron,
    RUN_DIR,
    cKDTree,
    download_buttons,
    mo,
    np,
    plt,
    plot_atoms,
    seed_choice,
    shape_choice,
    size_choice,
    write_structures,
):
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


    particle = build_particle(shape_choice.value, size_choice.value)
    seed = int(seed_choice.value)
    tag = f"{shape_choice.value}_{size_choice.value}nm_seed{seed}"          # used in every file name below
    write_structures(RUN_DIR / f"initial_{shape_choice.value}_{size_choice.value}nm.xyz", particle, format="extxyz")

    _x = particle.positions
    _neighbours = np.array([len(v) - 1 for v in cKDTree(_x).query_ball_point(_x, 3.0)])
    _surface = _neighbours < 12
    _d_equiv = (6 * len(particle) * CU_A0**3 / 4 / np.pi) ** (1 / 3) / 10
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    plot_atoms(particle, _ax, rotation="20x,35y,0z", show_unit_cell=0,
               radii=0.65, colors=np.where(_surface, "tab:blue", "goldenrod").tolist())
    _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.vstack([
            mo.md(f"""
    **{shape_choice.selected_key}**, seed {seed}

    - {len(particle)} atoms
    - equal-volume diameter {_d_equiv:.2f} nm
    - surface atoms (fewer than 12 neighbours, blue): {_surface.mean():.0%}
    """),
            download_buttons(f"initial_{shape_choice.value}_{size_choice.value}nm"),
        ]),
    ], justify="start", align="center")

    return particle, seed, tag


@app.cell(hide_code=True)
def exp1_md(mo):
    mo.md(r"""
    ## Experiment 1 · Does the energy become stable?

    The particle starts as a perfect crystal with all atoms at rest. A Langevin thermostat then holds it at a fixed
    temperature for up to 100 ps (2 fs time steps). The notebook records, every 0.2 ps:

    - the potential energy per atom, $E_\mathrm{pot}/N$;
    - the kinetic energy per atom, $E_\mathrm{kin}/N$, and the instantaneous temperature $T = 2E_\mathrm{kin}/(3Nk_\mathrm{B})$.
    """)
    return


@app.cell
def exp1_controls(ARGS, mo):
    stability_T = mo.ui.slider(300, 1300, step=50, value=int(ARGS.get("temperature", 900)),
                              label="Temperature (K)", show_value=True)
    _lengths = {f"{v} ps": v for v in sorted({20, 50, 100, int(ARGS.get("length", 20))})}
    stability_ps = mo.ui.dropdown(_lengths, value=f"{int(ARGS.get('length', 20))} ps", label="Length")
    run_stability = mo.ui.run_button(label="Run at fixed temperature")
    mo.hstack([stability_T, stability_ps, run_stability], justify="start", gap=2)

    return run_stability, stability_T, stability_ps


@app.cell
def exp1_run(
    download_buttons,
    experiment,
    mo,
    particle,
    run_md,
    run_stability,
    script_runs,
    seed,
    stability_T,
    stability_ps,
    tag,
):
    stability_name = f"stability_{tag}_T{stability_T.value}_{stability_ps.value}ps"
    stability = experiment(
        stability_name,
        lambda: run_md(particle, [("hold", float(stability_T.value), 0.2)] * (5 * stability_ps.value), seed,
                       title=f"{stability_ps.value} ps at {stability_T.value} K", start_at_rest=True),
        run_stability.value or script_runs("stability"),
    )
    (mo.md("Choose a temperature and a length, then press **Run at fixed temperature**.")
     if stability is None else download_buttons(stability_name))

    return (stability,)


@app.cell
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
def exp2_md(mo):
    mo.md(r"""
    ## Experiment 2 · Heating and cooling

    One run gives both the melting and the freezing of the particle:

    - **equilibrate:** 10 ps at 400 K;
    - **heat:** the thermostat temperature rises from 400 K to 1600 K at the chosen rate, in steps of 1 K;
    - **hold:** 20 ps at 1600 K; inspect the structures to establish whether melting occurred;
    - **cool:** back to 400 K at the same rate.

    One row of data is recorded at every kelvin. The `stage` column of the `.csv` file marks the four parts.
    By default, structures are saved every **100 K** on heating and cooling, plus the initial structure
    and the end of each stage: **27 snapshots per cycle**, independent of heating rate.
    Choose 50 K or 200 K spacing if you need more or fewer structures.

    Start at 10 K/ps to check the workflow. A slower rate increases the compute time, but it does not
    increase the number of CSV rows or snapshots.
    """)
    return


@app.cell
def exp2_controls(ARGS, mo):
    RATES = {"10 K/ps": 10.0, "5 K/ps": 5.0, "2 K/ps": 2.0, "1 K/ps": 1.0,
             "0.5 K/ps": 0.5, "0.2 K/ps": 0.2, "0.1 K/ps": 0.1}
    def cycle_schedule(rate, t_low=400.0, t_high=1600.0, hold_ps=20.0, equilibrate_ps=10.0):
        """(stage, thermostat temperature in K, length in ps) for every piece of the run."""
        steps = int(round(t_high - t_low))
        return ([("equilibrate", t_low, 1.0)] * int(equilibrate_ps)
                + [("heat", t_low + k + 1, 1.0 / rate) for k in range(steps)]
                + [("hold", t_high, 1.0)] * int(hold_ps)
                + [("cool", t_high - k - 1, 1.0 / rate) for k in range(steps)])


    _rate = float(ARGS.get("rate", 10.0))
    RATES.setdefault(f"{_rate:g} K/ps", _rate)
    cycle_rate = mo.ui.dropdown(RATES, value=f"{_rate:g} K/ps", label="Heating and cooling rate")
    snapshot_spacing = mo.ui.dropdown({"50 K": 50, "100 K": 100, "200 K": 200},
                                     value=f"{int(ARGS.get('snapshots', 100))} K", label="Snapshot spacing")
    run_cycle = mo.ui.run_button(label="Run heating and cooling (400 -> 1600 -> 400 K)")
    mo.hstack([cycle_rate, snapshot_spacing, run_cycle], justify="start", gap=2)

    return cycle_rate, cycle_schedule, run_cycle, snapshot_spacing


@app.cell
def exp2_run(
    cycle_rate,
    cycle_schedule,
    download_buttons,
    experiment,
    mo,
    particle,
    run_cycle,
    run_md,
    script_runs,
    seed,
    snapshot_spacing,
    tag,
):
    cycle_name = f"cycle_{tag}_{cycle_rate.value:g}Kps"
    cycle = experiment(
        cycle_name,
        lambda: run_md(particle, cycle_schedule(cycle_rate.value), seed,
                       title=f"Heating and cooling at {cycle_rate.value:g} K/ps", snapshot_kelvin=snapshot_spacing.value),
        run_cycle.value or script_runs("cycle"),
    )
    _ns = sum(piece[2] for piece in cycle_schedule(cycle_rate.value)) / 1000
    (mo.md(f"Press **Run heating and cooling**. At {cycle_rate.value:g} K/ps this is {_ns:.2g} ns of "
           f"simulation. The CSV contains 2430 rows. Snapshot spacing: {snapshot_spacing.value} K.")
     if cycle is None else download_buttons(cycle_name))

    return (cycle,)


@app.cell
def exp2_plot(cycle, cycle_rate, mo, particle, plt, seed, shape_choice):
    mo.stop(cycle is None)
    _fig, _ax = plt.subplots(figsize=(6, 3.8))
    for _stage, _color in (("heat", "tab:red"), ("hold", "tab:purple"), ("cool", "tab:blue")):
        _k = cycle["stage"] == _stage
        _ax.plot(cycle["T_set_K"][_k], cycle["PE_eV_per_atom"][_k], lw=1, color=_color, label=_stage)
    _ax.set(xlabel="Thermostat temperature (K)", ylabel="Potential energy (eV/atom)",
            title=f"{shape_choice.selected_key}, {len(particle)} atoms, {cycle_rate.value:g} K/ps, seed {seed}")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig

    return


@app.cell(hide_code=True)
def viewer_md(mo):
    mo.md(r"""
    ## Look inside your runs

    Choose a saved structure file and a frame. ASE shows a 4 Å-thick central slice.
    Compare the ordered low-temperature lattice with a high-temperature structure.
    """)
    return


@app.cell
def viewer_controls(RUN_DIR, cycle, cycle_rate, mo, stability, tag):
    # Re-list the files after each experiment finishes and select this particle's cycle when available.
    _ = (stability, cycle)
    _xyz_files = sorted(p.name for p in RUN_DIR.glob("*.xyz"))
    _preferred = f"cycle_{tag}_{cycle_rate.value:g}Kps.xyz"
    _initial = f"initial_{tag.rsplit('_seed', 1)[0]}.xyz"
    _selected = _preferred if _preferred in _xyz_files else _initial
    viewer_file = mo.ui.dropdown(_xyz_files, value=_selected, label="Structure file")
    viewer_file

    return (viewer_file,)


@app.cell
def viewer_frames(RUN_DIR, mo, read_structures, viewer_file):
    mo.stop(viewer_file.value is None, mo.md("Run an experiment first."))
    saved_frames = read_structures(RUN_DIR / viewer_file.value, index=":")
    saved_frame = mo.ui.slider(0, max(1, len(saved_frames) - 1), value=len(saved_frames) - 1,
                               disabled=len(saved_frames) == 1, label="Frame", show_value=True)
    mo.vstack([mo.md(f"`{viewer_file.value}`: {len(saved_frames)} frames, {len(saved_frames[0])} atoms"), saved_frame])
    return saved_frame, saved_frames


@app.cell
def viewer(np, plot_atoms, plt, saved_frame, saved_frames):
    _frame = saved_frames[saved_frame.value]
    _xyz = _frame.positions - _frame.positions.mean(axis=0)
    _slice = np.abs(_xyz[:, 2]) < 2.0
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    _atoms = _frame[_slice]
    _atoms.positions = _xyz[_slice]
    plot_atoms(_atoms, _ax, rotation="0x,0y,0z", show_unit_cell=0, radii=0.55)
    _ax.set_title(f"{_frame.info.get('stage', '')}, {_frame.info.get('T_set_K', '')} K")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()

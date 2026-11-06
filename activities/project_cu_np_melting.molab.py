# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ase==3.29.0",
#     "matplotlib==3.11.2",
#     "numpy==2.5.4",
#     "nvidia-cuda-crt==13.2.86",
#     "nvidia-cuda-nvcc==13.2.86",
#     "nvidia-cuda-nvrtc==13.2.86",
#     "nvidia-nvvm==13.2.86",
#     "openmm[cuda13]==8.6.1",
#     "scipy==1.18.1",
#     "weas-widget==0.2.6",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


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

    In molab, sign in and choose **Fork** to create an editable copy. Select a GPU machine for the
    simulation. Download each result before changing its settings: running the same settings again
    replaces files with the same names.
    """)
    return


@app.cell
def imports():
    import csv
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import openmm as mm
    import openmm.unit as unit
    from pathlib import Path
    from ase import Atoms
    from ase.cluster import Decahedron, Icosahedron, Octahedron
    from ase.io import read as read_structures, write as write_structures
    from scipy.spatial import cKDTree

    # Run without the browser, for example:
    #   python notebook.py --shape icosahedron --seed 2 --rate 1 --run cycle
    #   python notebook.py --temperature 1100 --length 100 --run stability,cycle
    # Options: --shape --size --seed --temperature --length --rate --run (stability, cycle, or both).
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
        from urllib.request import urlopen
        with urlopen("https://raw.githubusercontent.com/tiangroup-uofa/mate374-cmpt-methods-mate/main/activities/Cu_mishin1.eam.alloy", timeout=30) as response:
            POTENTIAL.write_bytes(response.read())

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
        read_structures,
        script_runs,
        unit,
        write_structures,
    )


@app.cell
def potential(POTENTIAL, mm, mo, np):
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
        """Use the GPU (CUDA) when it works; otherwise fall back to the much slower CPU."""
        for name, properties in (("CUDA", {"Precision": "mixed"}), ("CPU", {})):
            try:
                platform = mm.Platform.getPlatformByName(name)
                mm.Context(eam_system(2), mm.VerletIntegrator(0.001), platform, properties)
                return name, platform, properties
            except Exception:
                continue
        raise RuntimeError("OpenMM found neither a CUDA nor a CPU platform.")


    PLATFORM_NAME, PLATFORM, PLATFORM_PROPERTIES = pick_platform()
    mo.md(f"OpenMM {mm.__version__} runs on **{PLATFORM_NAME}**."
          + ("" if PLATFORM_NAME == "CUDA" else " No GPU found: choose a GPU machine, or each run takes hours."))

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
    unit,
    write_structures,
):
    COLUMNS = ["time_ps", "stage", "T_set_K", "T_K", "PE_eV_per_atom", "KE_eV_per_atom"]


    def run_md(atoms, schedule, seed, title, start_at_rest=False, frame_every_ps=None,
               dt_fs=2.0, friction_per_ps=2.0):
        """Langevin MD of a free Cu particle.

        schedule: list of (stage, set temperature in K, length in ps). The thermostat temperature
        is updated at the start of each piece, and one row of data is recorded at its end.
        Returns the data table and, if frame_every_ps is set, a list of structures (ASE Atoms).
        """
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
        frames = [frame(atoms.positions, 0.0, schedule[0][0], schedule[0][1])] if frame_every_ps else []
        time_ps, next_frame = 0.0, frame_every_ps or np.inf
        for stage, t_set, length_ps in mo.status.progress_bar(schedule, title=title, remove_on_exit=True):
            integrator.setTemperature(t_set * unit.kelvin)
            steps = max(1, round(length_ps * 1000 / dt_fs))
            integrator.step(steps)
            time_ps += steps * dt_fs / 1000
            want_frame = time_ps >= next_frame - 1e-9
            state = context.getState(getEnergy=True, getPositions=want_frame)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / EV_TO_KJMOL / n
            ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / EV_TO_KJMOL / n
            for c, v in zip(COLUMNS, (time_ps, stage, t_set, 2 * ke / (3 * KB_EV), pe, ke)):
                table[c].append(v)
            if want_frame:
                frames.append(frame(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                                    time_ps, stage, t_set))
                next_frame += frame_every_ps
        del context
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
    size_choice = mo.ui.dropdown(SIZES, value=f"{int(ARGS.get('size', 4))} nm", label="Diameter")
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
    _fig = plt.figure(figsize=(4.2, 4.2))
    _ax = _fig.add_subplot(projection="3d")
    _ax.scatter(*_x[~_surface].T, s=6, c="goldenrod", depthshade=True)
    _ax.scatter(*_x[_surface].T, s=6, c="tab:blue", depthshade=True)
    _ax.set_box_aspect((1, 1, 1))
    _ax.set_axis_off()
    _ax.view_init(elev=20, azim=35)
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
    _lengths = {f"{v} ps": v for v in sorted({20, 50, 100, int(ARGS.get("length", 100))})}
    stability_ps = mo.ui.dropdown(_lengths, value=f"{int(ARGS.get('length', 100))} ps", label="Length")
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
    - **hold:** 20 ps at 1600 K, where the particle is liquid;
    - **cool:** back to 400 K at the same rate.

    One row of data is recorded at every kelvin. The `stage` column of the `.csv` file marks the four parts.
    Structures are saved every 25 K in the `.xyz` file.

    Choose the rate in the menu. Before you press the button, the notebook shows the simulated time
    and the expected GPU time for that rate.
    """)
    return


@app.cell
def exp2_controls(ARGS, mo):
    RATES = {"10 K/ps": 10.0, "5 K/ps": 5.0, "2 K/ps": 2.0, "1 K/ps": 1.0,
             "0.5 K/ps": 0.5, "0.2 K/ps": 0.2, "0.1 K/ps": 0.1}
    MIN_PER_NS_4NM = 0.95   # measured on the molab GPU for about 2900 atoms


    def minutes_per_ns(n_atoms):
        """Rough GPU time per simulated ns: 0.95 min at 2900 atoms, about 1.6 min at 23 000 atoms."""
        return MIN_PER_NS_4NM * (0.9 + 0.1 * n_atoms / 2900)


    def cycle_schedule(rate, t_low=400.0, t_high=1600.0, hold_ps=20.0, equilibrate_ps=10.0):
        """(stage, thermostat temperature in K, length in ps) for every piece of the run."""
        steps = int(round(t_high - t_low))
        return ([("equilibrate", t_low, 1.0)] * int(equilibrate_ps)
                + [("heat", t_low + k + 1, 1.0 / rate) for k in range(steps)]
                + [("hold", t_high, 1.0)] * int(hold_ps)
                + [("cool", t_high - k - 1, 1.0 / rate) for k in range(steps)])


    _rate = float(ARGS.get("rate", 1.0))
    RATES.setdefault(f"{_rate:g} K/ps", _rate)
    cycle_rate = mo.ui.dropdown(RATES, value=f"{_rate:g} K/ps", label="Heating and cooling rate")
    run_cycle = mo.ui.run_button(label="Run heating and cooling (400 -> 1600 -> 400 K)")
    mo.hstack([cycle_rate, run_cycle], justify="start", gap=2)

    return cycle_rate, cycle_schedule, minutes_per_ns, run_cycle


@app.cell
def exp2_run(
    cycle_rate,
    cycle_schedule,
    download_buttons,
    experiment,
    minutes_per_ns,
    mo,
    particle,
    run_cycle,
    run_md,
    script_runs,
    seed,
    tag,
):
    cycle_name = f"cycle_{tag}_{cycle_rate.value:g}Kps"
    cycle = experiment(
        cycle_name,
        lambda: run_md(particle, cycle_schedule(cycle_rate.value), seed,
                       title=f"Heating and cooling at {cycle_rate.value:g} K/ps", frame_every_ps=25.0 / cycle_rate.value),
        run_cycle.value or script_runs("cycle"),
    )
    _ns = sum(piece[2] for piece in cycle_schedule(cycle_rate.value)) / 1000
    (mo.md(f"Press **Run heating and cooling**. At {cycle_rate.value:g} K/ps this is {_ns:.2g} ns of "
           f"simulation, about {max(0.5, minutes_per_ns(len(particle)) * _ns):.1f} min on the GPU.")
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

    Choose a saved structure file. The viewer plays its frames: use the frame slider under the picture,
    drag to rotate, and scroll to zoom.
    """)
    return


@app.cell
def viewer_controls(RUN_DIR, cycle, mo, stability):
    # Re-list the files after each experiment finishes.
    _ = (stability, cycle)
    _xyz_files = sorted(p.name for p in RUN_DIR.glob("*.xyz"))
    viewer_file = mo.ui.dropdown(_xyz_files, value=_xyz_files[-1] if _xyz_files else None, label="Structure file")
    viewer_file

    return (viewer_file,)


@app.cell
def viewer(RUN_DIR, mo, read_structures, viewer_file):
    from weas_widget import WeasWidget

    mo.stop(viewer_file.value is None, mo.md("Run an experiment first."))
    _frames = read_structures(RUN_DIR / viewer_file.value, index=":")
    _gui = {  # keep the picture, the frame slider, and fullscreen; hide the extra panels
        "controls": {"enabled": False},
        "meshLegend": {"enabled": False},
        "atomLegend": {"enabled": False},
        "buttons": {"enabled": True, "fullscreen": True, "undo": False, "redo": False, "export": False,
                    "import": False, "measurement": False},
        "timeline": {"enabled": True},
    }
    _viewer = WeasWidget(viewerStyle={"width": "480px", "height": "360px"}, guiConfig=_gui)
    _viewer.from_ase(_frames)
    _viewer.avr.model_style = 0          # spheres only, no bonds
    mo.vstack([
        mo.md(f"`{viewer_file.value}`: {len(_frames)} frames, {len(_frames[0])} atoms"),
        mo.ui.anywidget(_viewer._widget),
    ])

    return


if __name__ == "__main__":
    app.run()

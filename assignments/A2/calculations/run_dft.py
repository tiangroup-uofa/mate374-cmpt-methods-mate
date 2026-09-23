#!/usr/bin/env python3
"""Reproduce the five Assignment 2 Ar20 DFT reference structures.

Run this script with the ordinary ASE VASP calculator.  VASP writes only to
TemporaryDirectory instances, which are removed after each calculation.  The
only persistent output is ar20-benchmark-ground-truth.extxyz.

Example on the office machine:

    VASP_PP_PATH=/path/to/vasp_pp \
    VASP_COMMAND='mpirun -np 8 /path/to/vasp_gam' \
    python run_dft.py

For a smoke test, set ``A2_DFT_LIMIT=1`` and send the result to a temporary
path with ``A2_DFT_OUTPUT=/tmp/ar20-smoke.extxyz``. The default limit is all
five structures. The completed office reference run took 1382.9 s
(23.0 min), including the isolated-atom reference; the script prints the
measured wall time for each new run. No VASP working directory is retained.

Alternatively set VASP_BINARY, VASP_MPI_LAUNCHER, and VASP_NPROCS and the
script constructs the command. ASE_VASP_COMMAND is also accepted.
"""
from __future__ import annotations

import json
import os
import time
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.vasp import Vasp
from ase.io import write
from ase.optimize import BFGS

ROOT = Path(__file__).resolve().parent
OUTPUT = Path(
    os.environ.get(
        "A2_DFT_OUTPUT", ROOT / "ar20-benchmark-ground-truth.extxyz"
    )
)

# Parameters selected from fit-ready-100.json.  The seed lattice uses the
# preceding 104-frame value so that this script reproduces the stored geometry.
FIT_EPSILON_EV = 0.013707773760502738
FIT_SIGMA_A = 3.385647921528616
SEED_SIGMA_A = 3.3853296987198065
SEED = 203740
PERTURBATIONS_A = (0.0, 0.10, 0.15, 0.20, 0.25)
N_ATOMS = 20
CELL_A = np.array([23.0, 23.0, 23.3])

# Fixed electronic settings used for the reference data.  NSW=0 is the normal
# ASE/VASP single-point setting; no ionic relaxation is performed.
VASP_SETTINGS = dict(
    xc="PBE",
    ivdw=12,
    encut=420.0,
    prec="Accurate",
    ediff=1e-6,
    nelm=120,
    algo="Normal",
    ismear=0,
    sigma=0.01,
    ispin=1,
    isym=0,
    lreal=False,
    lasph=True,
    addgrid=False,
    lwave=False,
    lcharg=False,
    kpts=(1, 1, 1),
    gamma=True,
    ncore=1,
    nsw=0,
    ibrion=-1,
    potim=0.0,
    isif=3,
)


def pair_distances(atoms: Atoms) -> np.ndarray:
    distances = atoms.get_all_distances(mic=False)
    return distances[np.triu_indices(len(atoms), 1)]


def fcc_seed() -> np.ndarray:
    """Return the 20 closest sites of a compact fcc seed."""
    nearest = 2 ** (1 / 6) * SEED_SIGMA_A
    lattice = nearest * np.sqrt(2.0)
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    points = np.array(
        [
            lattice * (np.array(cell, dtype=float) + b)
            for cell in product(range(-3, 4), repeat=3)
            for b in basis
        ]
    )
    order = sorted(
        range(len(points)),
        key=lambda i: (
            round(float(np.dot(points[i], points[i])), 10),
            tuple(np.round(points[i], 10)),
        ),
    )
    positions = points[order[:N_ATOMS]].copy()
    positions -= positions.mean(axis=0)
    return positions


def center_in_cell(atoms: Atoms) -> Atoms:
    atoms = atoms.copy()
    atoms.set_cell(np.diag(CELL_A))
    atoms.center()
    atoms.pbc = True
    return atoms


def make_structures() -> list[Atoms]:
    """Regenerate the five fixed near-equilibrium structures."""
    base = Atoms("Ar" * N_ATOMS, positions=fcc_seed(), pbc=False)
    base.calc = LennardJones(epsilon=FIT_EPSILON_EV, sigma=FIT_SIGMA_A, rc=20.0)
    optimizer = BFGS(base, logfile=None)
    optimizer.run(fmax=1e-5, steps=1000)
    relaxed = base.positions.copy()
    relaxed -= relaxed.mean(axis=0)

    structures = []
    for index, amplitude in enumerate(PERTURBATIONS_A, start=1):
        rng = np.random.default_rng(SEED + index)
        if amplitude == 0.0:
            positions = relaxed.copy()
        else:
            for _ in range(10_000):
                positions = relaxed + rng.normal(
                    scale=amplitude, size=relaxed.shape
                )
                positions -= positions.mean(axis=0)
                candidate = Atoms("Ar" * N_ATOMS, positions=positions)
                if pair_distances(candidate).min() >= 3.2:
                    break
            else:
                raise RuntimeError(f"Could not make Ar20 variant {index}")

        atoms = center_in_cell(
            Atoms("Ar" * N_ATOMS, positions=positions, pbc=True)
        )
        distances = pair_distances(atoms)
        atoms.info.update(
            config_id=f"ar20_benchmark_{index:02d}",
            role="held-out-test",
            subset="ar20-benchmark",
            subset_index=index,
            geometry_method=(
                "LJ-relaxed fcc20 with fixed near-equilibrium perturbation"
            ),
            geometry_seed=str(SEED + index),
            perturbation_rms_A=float(amplitude),
            min_distance_A=float(distances.min()),
            max_distance_A=float(distances.max()),
            minimum_face_vacuum_A=float(
                min(atoms.positions.min(axis=0).min(),
                    (atoms.cell.lengths() - atoms.positions.max(axis=0)).min())
            ),
        )
        structures.append(atoms)
    return structures


def vasp_command() -> str:
    command = os.environ.get("ASE_VASP_COMMAND") or os.environ.get("VASP_COMMAND")
    if command:
        return command
    binary = os.environ.get("VASP_BINARY")
    launcher = os.environ.get("VASP_MPI_LAUNCHER", "mpirun")
    ranks = os.environ.get("VASP_NPROCS", "8")
    if not binary:
        raise RuntimeError(
            "Set ASE_VASP_COMMAND/VASP_COMMAND, or set VASP_BINARY "
            "with VASP_MPI_LAUNCHER and VASP_NPROCS"
        )
    return f"{launcher} -np {ranks} {binary}"


def calculate(atoms: Atoms, command: str, label: str):
    """Run one ordinary ASE/VASP single point in a disposable directory."""
    with TemporaryDirectory(prefix=f"a2-{label}-") as directory:
        calculator = Vasp(
            command=command,
            directory=directory,
            txt="vasp.out",
            **VASP_SETTINGS,
        )
        atoms = atoms.copy()
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces())
        return energy, forces


def main() -> None:
    if not os.environ.get("VASP_PP_PATH"):
        raise RuntimeError("VASP_PP_PATH must point to the PAW potential root")
    command = vasp_command()
    limit = int(os.environ.get("A2_DFT_LIMIT", str(len(PERTURBATIONS_A))))
    if not 1 <= limit <= len(PERTURBATIONS_A):
        raise ValueError(
            f"A2_DFT_LIMIT must be between 1 and {len(PERTURBATIONS_A)}"
        )
    structures = make_structures()[:limit]
    started = time.perf_counter()

    # The atom reference uses exactly the same cell as the Ar20 structures.
    reference = Atoms(
        "Ar", positions=[0.5 * CELL_A], cell=np.diag(CELL_A), pbc=True
    )
    isolated_energy, _ = calculate(reference, command, "reference")

    output_frames = []
    for atoms in structures:
        cluster_energy, forces = calculate(
            atoms, command, atoms.info["config_id"]
        )
        interaction = cluster_energy - N_ATOMS * isolated_energy
        frame = atoms.copy()
        frame.info.update(
            method="PBE-D3(BJ)",
            encut_eV=420.0,
            ediff_eV=1e-6,
            energy_kind="VASP_extrapolated_zero_smearing",
            cluster_total_energy_eV=cluster_energy,
            isolated_Ar_energy_eV=isolated_energy,
            interaction_energy_eV=interaction,
            interaction_energy_per_atom_eV=interaction / N_ATOMS,
        )
        frame.calc = SinglePointCalculator(
            frame, energy=cluster_energy, forces=forces
        )
        output_frames.append(frame)

    temporary_output = OUTPUT.with_suffix(".tmp.extxyz")
    write(temporary_output, output_frames, format="extxyz")
    temporary_output.replace(OUTPUT)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "frames": len(output_frames),
                "isolated_Ar_energy_eV": isolated_energy,
                "elapsed_seconds": time.perf_counter() - started,
                "command": command,
                "settings": VASP_SETTINGS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

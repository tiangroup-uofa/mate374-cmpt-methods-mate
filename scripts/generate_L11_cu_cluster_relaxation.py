#!/usr/bin/env python3
"""Generate the L11 Cu-cluster relaxation GIF and its static PDF fallback."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ase.calculators.emt import EMT
from ase.cluster.icosahedron import Icosahedron
from ase.optimize import FIRE
from ase.visualize.plot import plot_atoms


ROOT = Path(__file__).resolve().parents[1]
GIF_PATH = ROOT / "assets" / "L11-Cu-cluster-relaxation.gif"
PNG_PATH = ROOT / "assets" / "L11-Cu-cluster-relaxation.png"
N_ATOMS = 20
SEED = 374
FRAME_COUNT = 25
FRAME_DURATION_MS = 120


def initial_cluster():
    """Make a perturbed 20-atom fragment of a Cu icosahedron."""
    seed_cluster = Icosahedron("Cu", 3)
    center = seed_cluster.positions.mean(axis=0)
    nearest = np.argsort(np.linalg.norm(seed_cluster.positions - center, axis=1))[:N_ATOMS]
    atoms = seed_cluster[nearest]
    atoms.positions -= atoms.positions.mean(axis=0)
    atoms.positions += np.random.default_rng(SEED).normal(
        loc=0.0, scale=0.32, size=atoms.positions.shape
    )
    atoms.calc = EMT()
    return atoms


def make_frame(atoms, energy_excess, max_partial):
    """Draw one ASE-projected structure and its two relaxation diagnostics."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=300)
    fig.patch.set_facecolor("white")
    plot_atoms(
        atoms,
        ax=ax,
        rotation="15x,20y,0z",
        show_unit_cell=0,
        radii=0.72,
    )
    fig.suptitle(
        rf"$E - E_{{\min}} = {energy_excess:.3f}\,\mathrm{{eV}}$" "\n"
        rf"$\mathrm{{max}}_j |\partial E/\partial x_j| = "
        rf"{max_partial:.3f}\,\mathrm{{eV}}/\mathrm{{\AA}}$",
        fontsize=12,
        y=0.96,
    )
    fig.text(
        0.5,
        0.035,
        "20-atom Cu cluster · EMT",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.06, right=0.94)
    return fig


def main():
    atoms = initial_cluster()
    snapshots = []

    def record_snapshot():
        gradient = -atoms.get_forces()
        snapshots.append(
            (
                atoms.copy(),
                atoms.get_potential_energy(),
                float(np.max(np.abs(gradient))),
            )
        )

    record_snapshot()
    optimizer = FIRE(atoms, logfile=None, dt=0.05, maxstep=0.15)
    optimizer.attach(record_snapshot, interval=1)
    converged = optimizer.run(fmax=0.03, steps=400)
    if not converged:
        raise RuntimeError("FIRE did not converge within 400 steps")

    minimum_energy = snapshots[-1][1]
    selected = np.unique(
        np.rint(np.linspace(0, len(snapshots) - 1, FRAME_COUNT)).astype(int)
    )

    frames = []
    for index in selected:
        configuration, energy, max_partial = snapshots[index]
        fig = make_frame(configuration, energy - minimum_energy, max_partial)
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        frames.append(image)
        if index == selected[-1]:
            fig.savefig(PNG_PATH, dpi=300, facecolor="white")
        plt.close(fig)

    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(
        f"Relaxed {N_ATOMS} Cu atoms in {len(snapshots) - 1} FIRE steps; "
        f"E_min = {minimum_energy:.6f} eV; "
        f"max |∂E/∂qᵢ| = {snapshots[-1][2]:.4f} eV/Å"
    )
    print(f"Wrote {GIF_PATH.relative_to(ROOT)}")
    print(f"Wrote {PNG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

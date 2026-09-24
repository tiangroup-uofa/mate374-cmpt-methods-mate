"""Generate static Q3 cluster figures and the reference parity plot.

Run from the repository root with ``uv run --locked python activities/a2_q3_figures.py``.
The five training examples use the same fixed random order as the Q3 demo.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assignments" / "A2" / "calculations"
FIGURE_DIR = ROOT / "assignments" / "A2" / "figures"
ANSWER_FIGURE = ROOT / "answer-keys" / "A2" / "cluster-parity.png"

# Parameters from the fit using all 100 Ar3-Ar7 training clusters.
SIGMA_A = 3.385647921528616
EPSILON_EV = 0.013707773760502738
RANDOM_SEED = 374


def interaction_energy(atoms, sigma=SIGMA_A, epsilon=EPSILON_EV):
    positions = np.asarray(atoms.positions, dtype=float)
    i, j = np.triu_indices(len(positions), k=1)
    distances = np.linalg.norm(positions[i] - positions[j], axis=1)
    q = (sigma / distances) ** 6
    return float(np.sum(4 * epsilon * (q * q - q)))


def load_data():
    training = read(DATA_DIR / "dft-ground-truth.extxyz", index=":")
    held_out = read(DATA_DIR / "ar20-benchmark-ground-truth.extxyz", index=":")
    if len(training) != 100 or len(held_out) != 5:
        raise ValueError("Expected 100 Ar3-Ar7 training clusters and five Ar20 test clusters.")

    order = np.random.default_rng(RANDOM_SEED).permutation(len(training))
    examples = [training[index] for index in order[:5]]
    return training, held_out, examples


def _projected_positions(atoms):
    positions = np.asarray(atoms.positions, dtype=float)
    centered = positions - positions.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=True)
    projected = centered @ basis.T
    for column in range(3):
        pivot = np.argmax(np.abs(projected[:, column]))
        if projected[pivot, column] < 0:
            projected[:, column] *= -1
    return projected


def _draw_cluster(ax, atoms, *, marker_size, bond_cutoff):
    positions = _projected_positions(atoms)
    i, j = np.triu_indices(len(positions), k=1)
    distances = np.linalg.norm(positions[i] - positions[j], axis=1)
    for first, second in zip(i[distances < bond_cutoff], j[distances < bond_cutoff]):
        ax.plot(*positions[[first, second]].T, color="#8797a7", lw=0.8, alpha=0.7, zorder=1)
    ax.scatter(
        *positions.T,
        s=marker_size,
        color="#74a9cf",
        edgecolors="#174a70",
        linewidths=0.55,
        depthshade=True,
        zorder=2,
    )
    radius = max(float(np.max(np.ptp(positions, axis=0))) * 0.54, 1.3)
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=36)
    ax.set_axis_off()


def make_training_banner(examples):
    fig = plt.figure(figsize=(11, 2.25), layout="constrained")
    grid = fig.add_gridspec(2, 5, height_ratios=(2.1, 0.7))

    for index, atoms in enumerate(examples):
        structure = fig.add_subplot(grid[0, index], projection="3d")
        _draw_cluster(structure, atoms, marker_size=135, bond_cutoff=4.75)
        structure.set_title(f"{index + 1} · Ar$_{{{len(atoms)}}}$", fontsize=12, pad=1)

        qm = float(atoms.info["interaction_energy_eV"])
        lj = interaction_energy(atoms)
        energies = fig.add_subplot(grid[1, index])
        energies.axis("off")
        energies.text(
            0.5, 0.5, f"QM: {qm:+.5f} eV\nLJ: {lj:+.5f} eV",
            ha="center", va="center", fontsize=11, linespacing=1.3,
        )

    output = FIGURE_DIR / "q3-training-examples.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def make_ar20_banner(held_out):
    fig = plt.figure(figsize=(11, 1.35), layout="constrained")
    grid = fig.add_gridspec(1, 5)
    for index, atoms in enumerate(held_out):
        ax = fig.add_subplot(grid[0, index], projection="3d")
        _draw_cluster(ax, atoms, marker_size=48, bond_cutoff=4.2)
        ax.set_title(f"Ar$_{{20}}$ · {index + 1}", fontsize=11, pad=1)

    output = FIGURE_DIR / "q3-ar20-test-clusters.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def make_reference_parity(training, held_out):
    small_qm = np.array([float(atoms.info["interaction_energy_eV"]) for atoms in training])
    small_lj = np.array([interaction_energy(atoms) for atoms in training])
    large_qm = np.array([float(atoms.info["interaction_energy_eV"]) for atoms in held_out])
    large_lj = np.array([interaction_energy(atoms) for atoms in held_out])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    all_values = np.concatenate([small_qm, small_lj, large_qm, large_lj])
    pad = max(float(np.ptp(all_values)) * 0.06, 0.01)
    limits = (float(all_values.min()) - pad, float(all_values.max()) + pad)
    zoom_limits = (-0.15, 0.08)

    for ax in axes:
        ax.plot(limits, limits, "--", color="0.4", lw=0.9)
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")

    axes[0].scatter(small_qm, small_lj, label="100 Ar$_3$–Ar$_7$ clusters",
                    marker="o", color="#2676ad", s=20, alpha=0.85)
    axes[0].scatter(large_qm, large_lj, label="5 Ar$_{20}$ clusters",
                    marker="^", color="#e07a22", s=24, alpha=0.9)
    axes[0].set(
        xlim=limits,
        ylim=limits,
        xlabel="QM total interaction energy (eV)",
        ylabel="LJ total interaction energy (eV)",
        title="All training and test clusters",
    )
    axes[0].legend(fontsize=7, loc="upper left")

    zoom = (small_qm >= zoom_limits[0]) & (small_qm <= zoom_limits[1])
    axes[1].scatter(small_qm[zoom], small_lj[zoom], marker="o", color="#2676ad",
                    s=22, alpha=0.85)
    axes[1].plot(zoom_limits, zoom_limits, "--", color="0.4", lw=0.9)
    axes[1].set(
        xlim=zoom_limits,
        ylim=zoom_limits,
        xlabel="QM total interaction energy (eV)",
        ylabel="LJ total interaction energy (eV)",
        title="Zoom on the smaller-cluster energies",
    )

    ANSWER_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ANSWER_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ANSWER_FIGURE


def main():
    training, held_out, examples = load_data()
    outputs = (
        make_training_banner(examples),
        make_ar20_banner(held_out),
        make_reference_parity(training, held_out),
    )
    for output in outputs:
        print(f"Generated {output.relative_to(ROOT)}")
    print("Training examples:", ", ".join(atoms.info["config_id"] for atoms in examples))


if __name__ == "__main__":
    main()

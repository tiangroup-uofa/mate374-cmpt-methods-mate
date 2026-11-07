# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo==0.25.2", "numpy==2.5.4", "scipy==1.18.1", "matplotlib==3.11.2", "ase==3.29.0", "openmm[cuda12]==8.6.1"]
# ///
"""Instructor check of an extracted project package. No cloud services or widget CDNs.

uv run scripts/check_project_release.py --package PATH --output PRIVATE_PATH
Run the packaged MD notebook separately to measure GPU/CPU workloads without concurrency.
This check executes the analysis on real outputs, fits the bonus, and tests a private
copy of the simulation notebook with the exported tables inserted into its potential cell.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.cluster import Icosahedron
from ase.io import read, write
from scipy.spatial import cKDTree
import openmm as mm
import openmm.unit as unit


def notebook(path, overrides=None):
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = time.perf_counter()
    _, definitions = module.app.run(defs=overrides)
    return definitions, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package, output = args.package.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    md_file = package / "4-md-run/project_cu_np_melting.molab.py"
    analysis_file = package / "5-analysis/project_cu_np_analysis.molab.py"
    bonus_file = package / "6-bonus/project_bonus_eam_fit.molab.py"
    result = {"source_sha256": {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (md_file, analysis_file, bonus_file)}}
    plt.close("all")
    analysis, seconds = notebook(analysis_file)
    assert analysis["csv_names"] and analysis["xyz_names"], "Generate MD results before running this check."
    assert "frames" in analysis and "solid_like" in analysis
    result["analysis_seconds"] = seconds
    result["analysis_files"] = analysis["csv_names"] + analysis["xyz_names"]
    for number in plt.get_fignums():
        plt.figure(number).savefig(output / f"analysis-{number}.png", dpi=300)
    # Exercise every cycle, including both a small particle and a larger particle's g(r).
    for name in analysis["xyz_names"]:
        if not name.startswith("cycle"):
            continue
        texts = {p.name: p.read_text() for p in (package / "4-md-run/runs").glob("*.csv")}
        texts[name] = (package / "4-md-run/runs" / name).read_text()
        plt.close("all")
        current, elapsed = notebook(analysis_file, {"files": texts,
            "csv_names": sorted(n for n in texts if n.endswith(".csv")), "xyz_names": [name]})
        assert len(current["frames"]) == 27
        result.setdefault("analysis_per_cycle_seconds", {})[name] = elapsed
        for number in plt.get_fignums():
            plt.figure(number).savefig(output / f"{Path(name).stem}-analysis-{number}.png", dpi=300)
    plt.close("all")
    bonus, seconds = notebook(bonus_file)
    result["bonus"] = {"fit_seconds": seconds, "energy_mae_meV_atom": float(bonus["e_mae"]),
                       "force_mae_meV_A": float(bonus["f_mae"]),
                       "selected_penalty": float(bonus["alpha_best"])}
    for number in plt.get_fignums():
        plt.figure(number).savefig(output / f"bonus-fit-{number}.png", dpi=300)
    fitted_dir = output / "fitted-md"
    fitted_dir.mkdir(exist_ok=True)
    shutil.copy(package / "4-md-run/Cu_mishin1.eam.alloy", fitted_dir)
    shutil.copy(bonus["fitted_model_path"], fitted_dir)
    # This is the exact edit that a completed integration makes to the supplied notebook.
    source = md_file.read_text()
    old = "    eam = read_setfl(POTENTIAL)"
    assert source.count(old) == 1
    source = source.replace(old, '    with np.load(POTENTIAL.with_name("cu_fitted_eam.npz")) as fitted:\n'
                            '        eam = {key: fitted[key].copy() for key in ("phi", "density", "embed", "r_max", "rho_max", "cutoff")}')
    fitted_notebook = fitted_dir / "fitted_cu_md.py"
    fitted_notebook.write_text(source)
    plt.close("all")
    fitted, _ = notebook(fitted_notebook)
    result["bonus"]["platform"] = fitted["PLATFORM"].getName()
    feature_rows, coefficients = bonus["feature_rows"], bonus["coefficients"]
    atoms = Icosahedron("Cu", 3, latticeconstant=3.63)
    atoms.positions += np.random.default_rng(374).normal(0, 0.02, atoms.positions.shape)
    energy_features, force_features = feature_rows(atoms)
    exact_energy, exact_forces = energy_features @ coefficients, force_features @ coefficients
    finite_errors = []
    for atom, axis in ((0, 0), (10, 1), (30, 2), (54, 0)):
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[atom, axis] += 1e-5
        minus.positions[atom, axis] -= 1e-5
        fd = -(feature_rows(plus)[0] @ coefficients - feature_rows(minus)[0] @ coefficients) / 2e-5
        finite_errors.append(abs(fd - exact_forces[atom, axis]))
    integrator = mm.VerletIntegrator(0.002)
    context = mm.Context(fitted["eam_system"](len(atoms)), integrator,
                         fitted["PLATFORM"], fitted["PLATFORM_PROPERTIES"])
    context.setPositions(atoms.positions / 10)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / 96.48533212
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer) / 964.8533212
    del context, integrator
    result["bonus"].update(finite_difference_max_error_eV_A=float(max(finite_errors)),
        openmm_energy_error_eV_atom=float(abs(energy - exact_energy) / len(atoms)),
        openmm_force_max_error_eV_A=float(np.max(abs(forces - exact_forces))))
    assert max(finite_errors) < 1e-5
    # Mixed-precision GPU interpolation is less exact than the analytic CPU calculation.
    assert result["bonus"]["openmm_force_max_error_eV_A"] < 0.005
    particle = fitted["particle"]
    for temperature in (400, 900):
        table, frames = fitted["run_md"](particle, [("hold", temperature, .2)] * 100, 1,
                                         f"Fitted model, 20 ps at {temperature} K", snapshot_kelvin=100)
        fitted["save_csv"](fitted_dir / f"fitted-{temperature}K.csv", table)
        assert all(np.isfinite(table[key]).all() for key in table if key != "stage")
        result["bonus"][f"20ps_{temperature}K_final_T"] = float(table["T_K"][-1])
    table, frames = fitted["run_md"](particle, fitted["cycle_schedule"](10), 1,
                                     "Fitted model heating and cooling", snapshot_kelvin=100)
    fitted["save_csv"](fitted_dir / "fitted-cycle.csv", table)
    write(fitted_dir / "fitted-cycle.xyz", frames, format="extxyz")
    assert len(frames) == 27
    assert all(np.isfinite(table[key]).all() for key in table if key != "stage")
    max_density = 0.0
    for frame in frames:
        assert np.isfinite(frame.positions).all()
        distances = cKDTree(frame.positions).sparse_distance_matrix(cKDTree(frame.positions), 5.5, output_type="coo_matrix")
        use = distances.row != distances.col
        rho = np.bincount(distances.row[use], weights=(np.maximum(5.5 - distances.data[use], 0) / 5.5)**4, minlength=len(frame))
        max_density = max(max_density, float(rho.max()))
    reference_path = package / "4-md-run/runs/cycle_sphere_2nm_seed1_10Kps.xyz"
    reference_frames = read(reference_path, index=":")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, snapshots, csv_path in (("Mishin", reference_frames, reference_path.with_suffix(".csv")),
                                       ("Fitted Cu", frames, fitted_dir / "fitted-cycle.csv")):
        run = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        order = np.array([analysis["solid_like"](snapshot.positions).mean() for snapshot in snapshots])
        stages = np.array([snapshot.info["stage"] for snapshot in snapshots])
        temperatures = np.array([snapshot.info["T_set_K"] for snapshot in snapshots])
        for stage, style in (("heat", "-"), ("cool", ":")):
            mask = run["stage"] == stage
            axes[0].plot(run["T_set_K"][mask], run["PE_eV_per_atom"][mask] - run["PE_eV_per_atom"][0],
                         style, label=f"{label}, {stage}")
            mask = stages == stage
            axes[1].plot(temperatures[mask], order[mask], style, marker="o", ms=3, label=f"{label}, {stage}")
        result["bonus"][label + "_initial_solid_fraction"] = float(order[0])
        hot = stages == "hold"
        result["bonus"][label + "_hot_solid_fraction"] = float(order[hot][-1])
        result["bonus"][label + "_final_solid_fraction"] = float(order[-1])
    axes[0].set(xlabel="Thermostat temperature (K)", ylabel="Energy relative to initial 400 K (eV/atom)")
    axes[1].set(xlabel="Thermostat temperature (K)", ylabel="Solid-like fraction", ylim=(-.05, 1.05))
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "fitted-vs-mishin.png", dpi=300)
    result["bonus"]["max_density_in_cycle"] = max_density
    result["bonus"]["final_radius_A"] = float(np.linalg.norm(frames[-1].positions, axis=1).max())
    assert max_density < 10.0, "The trajectory left the embedding table range."
    # Confirm every supplied MD CSV/XYZ can be parsed and has bounded output.
    result["md_outputs"] = {}
    for path in sorted((package / "4-md-run/runs").glob("*.*")):
        entry = {"bytes": path.stat().st_size}
        if path.suffix == ".xyz":
            snapshots = read(path, index=":")
            entry.update(frames=len(snapshots), atoms=len(snapshots[0]))
            assert len(snapshots) <= 51
        elif path.suffix == ".csv":
            data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            entry["rows"] = len(data)
            assert len(data) <= 2430
            for field in data.dtype.names:
                if field != "stage":
                    assert np.isfinite(data[field]).all()
        result["md_outputs"][path.name] = entry
    (output / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

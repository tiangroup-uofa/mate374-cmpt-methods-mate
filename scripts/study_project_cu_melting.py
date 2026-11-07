# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo==0.25.2", "numpy==2.5.4", "scipy==1.18.1", "matplotlib==3.11.2", "ase==3.29.0", "openmm[cuda12]==8.6.1"]
# ///
"""Instructor study: refit the supplied Cu DFT data, then compare nanoparticle cycles.

uv run scripts/study_project_cu_melting.py --output instructor-private/project-demo/dft-melting-study
Default: 2 nm at 10 and 1 K/ps, 3 seeds each, both models; 4 nm at 1 K/ps, 1 seed, both models.
Runs sequentially. Completed cases are reused only when their input fingerprints match.
50 K structural sampling is used for this private study; the student default remains 100 K.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from ase.calculators.eam import EAM
from ase.io import read, write
from ase.visualize.plot import plot_atoms
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def potential_digest(path):
    if path.suffix != ".npz":
        return digest(path)
    h = hashlib.sha256()
    with np.load(path) as model:
        for key in sorted(model.files):
            h.update(key.encode())
            h.update(np.asarray(model[key]).tobytes())
    return h.hexdigest()


def load_notebook(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, definitions = module.app.run()
    plt.close("all")
    return definitions


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def crossing(temperatures, fraction, threshold, below=True):
    hit = fraction < threshold if below else fraction >= threshold
    for i in range(1, len(hit)):
        # Require the next sample to agree when there is one.
        if hit[i] and not hit[i - 1] and (i == len(hit) - 1 or hit[i + 1]):
            a, b = float(temperatures[i - 1]), float(temperatures[i])
            return [min(a, b), max(a, b)]
    return None


def analyze_case(folder, solid_like):
    data = np.genfromtxt(folder / "cycle.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    frames = read(folder / "cycle.xyz", index=":")
    assert all(np.isfinite(data[key]).all() for key in data.dtype.names if key != "stage")
    rows = []
    max_density, min_distance, max_radius = 0.0, np.inf, 0.0
    detached_counts = []
    for frame in frames:
        xyz = frame.positions - frame.positions.mean(axis=0)
        assert np.isfinite(xyz).all()
        radius = np.linalg.norm(xyz, axis=1)
        order = solid_like(xyz)
        tree = cKDTree(xyz)
        pairs = tree.query_pairs(3.8, output_type="ndarray")
        graph = coo_matrix((np.ones(2 * len(pairs)),
                            (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(len(xyz), len(xyz)))
        _, labels = connected_components(graph, directed=False)
        bound = labels == np.argmax(np.bincount(labels))
        detached_counts.append(int((~bound).sum()))
        centre = xyz[bound].mean(axis=0)
        bound_radius = np.linalg.norm(xyz - centre, axis=1)
        core = bound & (bound_radius < 0.5 * bound_radius[bound].max())
        if not core.any():
            core = bound
        rows.append((frame.info["stage"], frame.info["T_set_K"], order.mean(), order[core].mean()))
        sparse = tree.sparse_distance_matrix(tree, 5.5, output_type="coo_matrix")
        use = sparse.row != sparse.col
        min_distance = min(min_distance, float(sparse.data[use].min()))
        rho = np.bincount(sparse.row[use], weights=(np.maximum(5.5 - sparse.data[use], 0) / 5.5)**4, minlength=len(frame))
        max_density = max(max_density, float(rho.max()))
        max_radius = max(max_radius, float(radius.max()))
    stages = np.array([r[0] for r in rows])
    values = np.array([r[1:] for r in rows])
    baseline = values[stages == "equilibrate"][0, 1:]
    result = {"atoms": len(frames[0]), "snapshots": len(frames), "rows": len(data),
              "csv_bytes": (folder / "cycle.csv").stat().st_size,
              "xyz_bytes": (folder / "cycle.xyz").stat().st_size,
              "max_density": max_density, "min_separation_A": min_distance, "max_radius_A": max_radius,
              "hot_solid_fraction": float(values[stages == "hold"][-1, 1]),
              "final_solid_fraction": float(values[-1, 1]),
              "max_detached_atoms": max(detached_counts), "final_detached_atoms": detached_counts[-1],
              "hot_detached_atoms": int(np.array(detached_counts)[stages == "hold"][-1])}
    for name, column in (("whole", 1), ("core", 2)):
        initial = baseline[column - 1]
        heat = stages == "heat"
        cool = stages == "cool"
        result[name + "_equilibrated_solid_fraction"] = float(initial)
        for level in (.8, .5, .2):
            result[f"{name}_heat_{level:g}_bracket_K"] = crossing(values[heat, 0], values[heat, column], initial * level)
        result[name + "_freeze_0.5_bracket_K"] = crossing(values[cool, 0], values[cool, column], initial * .5, below=False)
    heat = data["stage"] == "heat"
    temperature = data["T_set_K"][heat]
    slope = savgol_filter(data["PE_eV_per_atom"][heat], 101, 3, deriv=1, delta=1.0)
    inside = (temperature >= 650) & (temperature <= 1500)
    result["energy_slope_peak_K"] = float(temperature[inside][np.argmax(slope[inside])])
    dump(folder / "order.json", [{"stage": r[0], "T_set_K": float(r[1]), "solid_fraction": float(r[2]),
                                  "core_fraction": float(r[3]), "detached_atoms": detached}
                                 for r, detached in zip(rows, detached_counts)])
    dump(folder / "observations.json", result)
    return result


def plot_summary(output, results, figures=None):
    figures = output if figures is None else figures
    figures.mkdir(parents=True, exist_ok=True)
    for size in sorted({r["diameter_nm"] for r in results}):
        rates = sorted({r["rate_K_ps"] for r in results if r["diameter_nm"] == size}, reverse=True)
        fig, axes = plt.subplots(len(rates), 2, figsize=(11, 4 * len(rates)), squeeze=False)
        for i, rate in enumerate(rates):
            for result in results:
                if result["diameter_nm"] != size or result["rate_K_ps"] != rate:
                    continue
                folder = output / result["case"]
                data = np.genfromtxt(folder / "cycle.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
                order = json.loads((folder / "order.json").read_text())
                colour = "tab:blue" if result["model"] == "mishin" else "tab:orange"
                for stage, style in (("heat", "-"), ("cool", ":")):
                    mask = data["stage"] == stage
                    label = f"{result['model']}, seed {result['seed']}, {stage}"
                    axes[i, 0].plot(data["T_set_K"][mask], data["PE_eV_per_atom"][mask] - data["PE_eV_per_atom"][0], style, color=colour, alpha=.65, lw=.8)
                    selected = [r for r in order if r["stage"] == stage]
                    axes[i, 1].plot([r["T_set_K"] for r in selected], [r["core_fraction"] for r in selected], style, color=colour, alpha=.65, lw=1, label=label)
            axes[i, 0].set(xlabel="Thermostat temperature (K)", ylabel="Energy above initial 400 K (eV/atom)", title=f"{size} nm sphere, {rate:g} K/ps")
            axes[i, 1].set(xlabel="Thermostat temperature (K)", ylabel="Core solid-like fraction", ylim=(-.05, 1.05))
            axes[i, 1].axhline(.5, color="grey", lw=.6)
            # Distinguish model and direction without 12 repeated seed labels.
            from matplotlib.lines import Line2D
            axes[i, 1].legend(handles=[Line2D([], [], color=c, ls=s, label=f"{m}, {d}")
                for m, c in (("Mishin", "tab:blue"), ("DFT fit", "tab:orange")) for d, s in (("heat", "-"), ("cool", ":"))], frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"melting-{size}nm.png", dpi=300)
        plt.close(fig)
    # ASE views of the matched 4 nm runs (or 2 nm while the study is still in progress).
    selected_size = max(r["diameter_nm"] for r in results)
    selected_rate = min(r["rate_K_ps"] for r in results if r["diameter_nm"] == selected_size)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for row, model in enumerate(("mishin", "fitted")):
        case = next((r for r in results if r["model"] == model and r["diameter_nm"] == selected_size and r["rate_K_ps"] == selected_rate and r["seed"] == 1), None)
        if case is None:
            continue
        frames = read(output / case["case"] / "cycle.xyz", index=":")
        heat = [f for f in frames if f.info["stage"] == "heat"]
        for col, temperature in enumerate((600, 1000, 1400)):
            frame = min(heat, key=lambda f: abs(f.info["T_set_K"] - temperature))
            positions = frame.positions - frame.positions.mean(axis=0)
            atoms = frame[np.abs(positions[:, 2]) < 2]
            plot_atoms(atoms, axes[row, col], show_unit_cell=0, rotation="0x,0y,0z", radii=.55)
            axes[row, col].set_title(f"{model}, {frame.info['T_set_K']:g} K")
    fig.suptitle(f"ASE central slices: {selected_size} nm, {selected_rate:g} K/ps, seed 1")
    fig.tight_layout()
    fig.savefig(figures / "ase-structures.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Run only the matched 2 nm, 10 K/ps, seed 1 pair.")
    parser.add_argument("--fit-only", action="store_true", help="Refit and check lattice energies without starting MD.")
    parser.add_argument("--reference-output", type=Path, help="Reuse completed Mishin cases with identical input fingerprints.")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "instructor-private"):
        parser.error("Keep completed bonus work and study outputs inside instructor-private/.")
    output.mkdir(parents=True, exist_ok=True)
    try:
        gpu_status = subprocess.check_output(["nvidia-smi"], text=True)
    except FileNotFoundError:
        gpu_status = "nvidia-smi is unavailable"
    (output / "gpu-at-start.txt").write_text(gpu_status)
    fit_dir = output / "fit"
    fit_dir.mkdir(exist_ok=True)
    for source in (ROOT / "activities/project_bonus_eam_fit.molab.py", ROOT / "project/data/cu_vasp_train.xyz", ROOT / "project/data/cu_vasp_test.xyz"):
        shutil.copy2(source, fit_dir)
    fit_started = time.perf_counter()
    bonus = load_notebook(fit_dir / "project_bonus_eam_fit.molab.py")
    fit_report = {"seconds": time.perf_counter() - fit_started, "train_count": len(bonus["train"]), "test_count": len(bonus["test"]),
                  "energy_mae_meV_atom": float(bonus["e_mae"]), "force_mae_meV_A": float(bonus["f_mae"]),
                  "ridge_penalty": float(bonus["alpha_best"]), "coefficients": bonus["coefficients"].tolist(),
                  "inputs_sha256": {p.name: digest(p) for p in fit_dir.glob("*.xyz")}}
    family_key = next((k for k in ("family", "config_type", "structure_type") if k in bonus["test"][0].info), None)
    fit_report["metadata_example"] = {k: str(v) for k, v in bonus["test"][0].info.items()}
    if family_key:
        for family in sorted({a.info[family_key] for a in bonus["test"]}):
            indices = [i for i, a in enumerate(bonus["test"]) if a.info[family_key] == family]
            energy_errors, force_errors = [], []
            for i in indices:
                a = bonus["test"][i]
                e, f = bonus["feature_rows"](a)
                energy_errors.append(abs(e @ bonus["coefficients"] - a.info["dft_energy"]) / len(a))
                force_errors.extend(abs(f @ bonus["coefficients"] - a.arrays["dft_forces"]).ravel())
            fit_report.setdefault("families", {})[str(family)] = {"count": len(indices), "energy_mae_meV_atom": float(np.mean(energy_errors) * 1000), "force_mae_meV_A": float(np.mean(force_errors) * 1000)}
    reference = EAM(potential=str(ROOT / "activities/Cu_mishin1.eam.alloy"))
    def bulk_energy(a, model):
        atoms = bulk("Cu", "fcc", a=a, cubic=True)
        if model == "fitted":
            return float(bonus["feature_rows"](atoms)[0] @ bonus["coefficients"] / len(atoms))
        atoms.calc = reference
        return float(atoms.get_potential_energy() / len(atoms))
    for model in ("mishin", "fitted"):
        optimum = minimize_scalar(lambda a: bulk_energy(a, model), bounds=(3.1, 4.2), method="bounded")
        fit_report.setdefault("fcc_lattice", {})[model] = {"a0_A": float(optimum.x), "energy_eV_atom": float(optimum.fun), "bounds_A": [3.1, 4.2]}
    dump(output / "fit-report.json", fit_report)
    print("FRESH FIT", json.dumps(fit_report), flush=True)
    if args.fit_only:
        return
    md_source = (ROOT / "activities/project_cu_np_melting.molab.py").read_text()
    md_source = md_source.replace('ARGS.get("platform", "Auto")', 'ARGS.get("platform", "CUDA")')
    namespaces = {}
    for model in ("mishin", "fitted"):
        folder = output / model
        folder.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "activities/Cu_mishin1.eam.alloy", folder)
        shutil.copy2(bonus["fitted_model_path"], folder)
        source = md_source
        if model == "fitted":
            old = "    eam = read_setfl(POTENTIAL)"
            assert source.count(old) == 1
            source = source.replace(old, '    with np.load(POTENTIAL.with_name("cu_fitted_eam.npz")) as fitted_tables:\n'
                                  '        eam = {key: fitted_tables[key].copy() for key in ("phi", "density", "embed", "r_max", "rho_max", "cutoff")}')
        path = folder / "completed_md.py"
        path.write_text(source)
        namespaces[model] = load_notebook(path)
        assert namespaces[model]["PLATFORM"].getName() == "CUDA"
    analysis = load_notebook(ROOT / "activities/project_cu_np_analysis.molab.py")
    cases = [(2, 10, seed) for seed in (1, 2, 3)] + [(2, 1, seed) for seed in (1, 2, 3)] + [(4, 1, 1)]
    if args.quick:
        cases = [(2, 10, 1)]
    results = []
    started = time.perf_counter()
    for diameter, rate, seed in cases:
        for model in ("mishin", "fitted"):
            tag = f"{model}-{diameter}nm-{rate:g}Kps-seed{seed}"
            folder = output / tag
            folder.mkdir(exist_ok=True)
            inputs = {"model": model, "diameter_nm": diameter, "rate_K_ps": rate, "seed": seed,
                      "snapshot_kelvin": 50, "dt_fs": 2, "friction_per_ps": 2, "platform": "CUDA", "precision": "mixed",
                      "potential_sha256": potential_digest(output / model / ("cu_fitted_eam.npz" if model == "fitted" else "Cu_mishin1.eam.alloy"))}
            manifest = folder / "inputs.json"
            if model == "mishin" and args.reference_output and not manifest.exists():
                previous = args.reference_output.resolve() / tag
                if (previous / "inputs.json").exists() and json.loads((previous / "inputs.json").read_text()) == inputs:
                    shutil.copytree(previous, folder, dirs_exist_ok=True)
            if manifest.exists() and (folder / "cycle.xyz").exists() and (folder / "cycle.csv").exists():
                if json.loads(manifest.read_text()) != inputs:
                    raise ValueError(f"Existing case has different inputs: {folder}")
                print("REUSE", tag, flush=True)
                seconds = json.loads((folder / "timing.json").read_text())["seconds"]
            else:
                print("START", tag, flush=True)
                ns = namespaces[model]
                atoms = ns["build_particle"]("sphere", diameter)
                case_started = time.perf_counter()
                table, frames = ns["run_md"](atoms, ns["cycle_schedule"](rate), seed, tag, snapshot_kelvin=50)
                seconds = time.perf_counter() - case_started
                ns["save_csv"](folder / "cycle.csv", table)
                write(folder / "cycle.xyz", frames, format="extxyz")
                dump(folder / "timing.json", {"seconds": seconds})
                dump(manifest, inputs)
            observations = analyze_case(folder, analysis["solid_like"])
            observations.update({k: inputs[k] for k in ("model", "diameter_nm", "rate_K_ps", "seed")})
            observations.update(seconds=seconds, case=tag)
            results.append(observations)
            dump(output / "results.json", results)
            print("RESULT", json.dumps(observations), flush=True)
            if (output / "STOP_AFTER_CASE").exists():
                plot_summary(output, results)
                print("STOP_AFTER_CASE requested; completed cases are saved.", flush=True)
                return
        plot_summary(output, results)
    dump(output / "study-summary.json", {"fit": fit_report, "cases": results, "driver_seconds_this_invocation": time.perf_counter() - started})
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()

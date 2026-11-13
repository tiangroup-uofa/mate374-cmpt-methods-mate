# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo==0.25.2", "numpy==2.5.4", "scipy==1.18.1", "matplotlib==3.11.2", "ase==3.29.0", "openmm==8.6.1", "plotly==7.1.0", "pypdf>=6"]
# ///
"""Maintainer smoke test for the current project release, using an extracted private copy.

uv run scripts/check_project_release.py --archive project/MATE374-group-project.zip \
    --output instructor-private/project-demo/release-check

Checks PDF hyperlinks and checksums, then exercises the notebooks and CSV/XYZ transfer.
Student exercise placeholders are deliberately preserved. Uses the installed compute platform;
this does not verify anonymous molab forks or provision a new cloud GPU environment.
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import textwrap
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
from ase import Atoms
from ase.io import read
from pypdf import PdfReader

REPOSITORY = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, default=REPOSITORY / "project/MATE374-group-project.zip")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ROOT = args.output.resolve()
if not ROOT.is_relative_to(REPOSITORY / "instructor-private"):
    parser.error("Keep verification outputs inside ignored instructor-private/.")
ROOT.mkdir(parents=True, exist_ok=True)
PACKAGE = ROOT / "extracted-check"
if PACKAGE.exists():
    parser.error("Use a fresh --output directory to avoid stale notebook data.")
with zipfile.ZipFile(args.archive) as archive:
    for name in archive.namelist():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Unsafe ZIP member: {name}")
    archive.extractall(PACKAGE)
EXPECTED = {
    "Simulation": "https://molab.marimo.io/notebooks/nb_snapzMa8oVN3beQpXGuBNc",
    "Analysis": "https://molab.marimo.io/notebooks/nb_2m8VYhDJC1A8y2MXkFXtUk",
    "Bonus": "https://molab.marimo.io/notebooks/nb_6yiNfKPqBZQSU8LNjCPHco",
}
report = {}


def notebook(path, overrides=None):
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start = time.perf_counter()
    _, definitions = module.app.run(defs=overrides)
    return definitions, time.perf_counter() - start


with zipfile.ZipFile(args.archive) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist()))
    for line in archive.read("SHA256SUMS.txt").decode().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    for filename in ("1-assignment.pdf", "2-assignment-tutorial.pdf"):
        reader = PdfReader(io.BytesIO(archive.read(filename)))
        links = []
        for number, page in enumerate(reader.pages, 1):
            for annotation in page.get("/Annots", []):
                action = annotation.get_object().get("/A")
                if action and action.get("/URI"):
                    links.append((number, str(action["/URI"])))
        report[filename] = {"pages": len(reader.pages), "notebook_link_pages": {}}
        for name, url in EXPECTED.items():
            pages = [number for number, target in links if target == url]
            assert pages, (filename, name)
            report[filename]["notebook_link_pages"][name] = pages
    assert "100 ps" in "".join(p.extract_text() for p in PdfReader(
        io.BytesIO(archive.read("2-assignment-tutorial.pdf"))).pages)
    for folder, source in (
        ("4-md-run", "project_cu_np_melting.molab.py"),
        ("5-analysis", "project_cu_np_analysis.molab.py"),
        ("6-bonus", "project_bonus_eam_fit.molab.py"),
    ):
        assert (PACKAGE / folder / source).read_bytes() == (
            REPOSITORY / "activities" / source).read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive.read("7-report-template-source.zip"))) as template:
        assert set(template.namelist()) == {
            "report.tex", "references.bib", "fig_stability.png", "fig_cycle.png", "fig_size.png"}
        assert "../../references" not in template.read("report.tex").decode()
report["zip_manifest"] = "all entries match SHA256SUMS.txt"

md_path = PACKAGE / "4-md-run/project_cu_np_melting.molab.py"
md, seconds = notebook(md_path)
assert md["potential_ready"] and md["cycle"] is None
schedule = md["cycle_schedule"](0.1, t_low=950, t_high=951, hold_ps=1)
cluster = Atoms("Cu4", positions=[[0, 0, 0], [2.56, 0, 0], [1.28, 2.2, 0], [1.28, 0.7, 2.1]])
table = md["experiment"](
    "release-validation",
    lambda: md["run_md"](cluster, schedule, 1, "Extracted release validation",
                         snapshot_kelvin=1, equilibration_ps=100, equilibration_temperature=950),
    True,
)
run_dir = PACKAGE / "4-md-run/runs"
eq = md["load_csv"](run_dir / "release-validation_equilibration.csv")
frames = read(run_dir / "release-validation.xyz", index=":")
assert len(eq["time_ps"]) == 500 and abs(eq["time_ps"][-1] - 100) < 1e-8
np.testing.assert_allclose(eq["T_set_K"], 950)
assert set(table["stage"]) == {"heat", "hold", "cool"}
np.testing.assert_allclose(table["time_ps"], [10, 11, 21])
np.testing.assert_allclose([a.info["time_ps"] for a in frames], [0, 10, 11, 21])
assert all(a.info["stage"] != "equilibrate" for a in frames)
assert frames[0].info["T_set_K"] == 950
for key in table:
    if key != "stage":
        assert np.isfinite(table[key]).all()
assert all(np.isfinite(a.positions).all() for a in frames)
report["md"] = {
    "platform": md["PLATFORM_NAME"], "startup_seconds": seconds,
    "equilibration_records": len(eq["time_ps"]), "cycle_records": len(table["time_ps"]),
    "xyz_frames": len(frames), "cycle_times_ps": table["time_ps"].tolist(),
}
print("MD and separate equilibration export: PASS", flush=True)

bonus_path = PACKAGE / "6-bonus/project_bonus_eam_fit.molab.py"
bonus, seconds = notebook(bonus_path)
assert bonus["fit_result"].success
assert len(bonus["coefficients"]) == 15
assert np.isfinite(bonus["coefficients"]).all()
assert len(bonus["reference_axes"][2].patches) == 2
assert all(p.get_alpha() == 0.5 for p in bonus["reference_axes"][2].patches)
source = bonus_path.read_text()
copy_cell = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "copy_function")
builder_node = next(n for n in copy_cell.body if isinstance(n, ast.FunctionDef) and n.name == "custom_system")
builder_source = textwrap.dedent(ast.get_source_segment(source, builder_node))
assert "coefficients = np.array([], dtype=float)" in builder_source
completed_source = builder_source.replace(
    "coefficients = np.array([], dtype=float)",
    "coefficients = np.array(" + repr(bonus["coefficients"].tolist()) + ", dtype=float)")
namespace = {"np": np, "mm": bonus["mm"]}
exec(completed_source, namespace)
builder = namespace["custom_system"]
features, force_features = bonus["feature_rows"](cluster)
energy, forces = bonus["openmm_energy_forces"](builder(len(cluster)), cluster.positions)
energy_error = abs(float(energy - features @ bonus["coefficients"]))
force_error = float(np.max(np.abs(forces - force_features @ bonus["coefficients"])))
assert energy_error < 1e-5 and force_error < 5e-3
report["bonus"] = {
    "seconds": seconds, "training_frames": len(bonus["fit_structures"]),
    "heldout_frames": len(bonus["heldout_structures"]),
    "energy_mae_eV_atom": float(bonus["energy_mae"]),
    "force_mae_eV_A": float(bonus["force_mae"]),
    "copied_function_energy_error_eV": energy_error,
    "copied_function_force_error_eV_A": force_error,
    "student_coefficient_array_remains_empty": True,
}
print("Bonus fit, required inputs and completed-function check: PASS", flush=True)

csv_upload = SimpleNamespace(value=[SimpleNamespace(
    name="release-validation.csv", contents=(run_dir / "release-validation.csv").read_bytes())])
xyz_upload = SimpleNamespace(value=[SimpleNamespace(
    name="release-validation.xyz", contents=(run_dir / "release-validation.xyz").read_bytes())])
analysis, seconds = notebook(PACKAGE / "5-analysis/project_cu_np_analysis.molab.py",
                             {"csv_upload": csv_upload, "xyz_upload": xyz_upload})
assert len(analysis["frames"]) == 4
assert len(analysis["csv_data"]) == 3
assert analysis["fit_melting_data"](np.array([2., 3., 4.]), np.array([900., 1000., 1100.])) is None
report["analysis"] = {"seconds": seconds, "csv_records": len(analysis["csv_data"]),
                      "xyz_frames": len(analysis["frames"]), "student_fit_exercise_preserved": True}
print("Analysis CSV/XYZ uploads and intended student exercise: PASS", flush=True)
(ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

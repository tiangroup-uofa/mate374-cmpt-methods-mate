#!/usr/bin/env python3
"""Build the private Canvas ZIP from an explicit file list.

Render instructor-private/project-demo/{assignment,tutorial}.qmd and the sample report
before running this script. Student names remain in ignored instructor-private/ only.
The source report bibliography is copied from the root references.bib at build time.
"""
import argparse
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "instructor-private/project-demo"


def report_sources():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        source = (ROOT / "project/report-template/report.tex").read_text()
        archive.writestr("report.tex", source.replace(r"\bibliography{../../references}", r"\bibliography{references}"))
        archive.write(ROOT / "references.bib", "references.bib")
        for name in ("fig_stability.png", "fig_cycle.png", "fig_size.png"):
            archive.write(ROOT / "project/report-template" / name, name)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIVATE / "MATE374-group-project.zip")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "instructor-private"):
        parser.error("The named Canvas package must stay inside ignored instructor-private/.")
    required = [PRIVATE / "assignment.pdf", PRIVATE / "tutorial.pdf",
                ROOT / "project/report-template/report.pdf"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Render this PDF first: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        # Keep the requested document order in the archive, independently of a file manager's sorting.
        for path, name in zip(required, ("1-assignment.pdf", "2-assignment-tutorial.pdf", "3-assignment-sample-report.pdf")):
            archive.write(path, name)
        for source, target in (
            ("activities/project_cu_np_melting.molab.py", "4-md-run/project_cu_np_melting.molab.py"),
            ("activities/Cu_mishin1.eam.alloy", "4-md-run/Cu_mishin1.eam.alloy"),
            ("activities/project_cu_np_analysis.molab.py", "5-analysis/project_cu_np_analysis.molab.py"),
            ("activities/project_bonus_eam_fit.molab.py", "6-bonus/project_bonus_eam_fit.molab.py"),
            ("project/data/cu_vasp_train.xyz", "6-bonus/cu_vasp_train.xyz"),
            ("project/data/cu_vasp_test.xyz", "6-bonus/cu_vasp_test.xyz"),
        ):
            archive.write(ROOT / source, target)
        for folder, notebook, description in (
            ("4-md-run", "project_cu_np_melting.molab.py", "Keep Cu_mishin1.eam.alloy beside the notebook. Results go into runs/. Select Auto, CUDA, or CPU in the notebook. CPU defaults to 4 threads. Start with a 2 nm sphere, 20 ps at 900 K, then a 10 K/ps cycle with 100 K snapshot spacing. The cycle saves 2430 CSV rows and 27 XYZ frames."),
            ("5-analysis", "project_cu_np_analysis.molab.py", "This notebook reads ../4-md-run/runs/ at startup. Use Run all to refresh after a new simulation, or upload CSV/XYZ files. You can also put files in a runs/ folder beside this notebook. ASE draws the atomic structures through its matplotlib interface. The notebook needs no external 3D widget."),
            ("6-bonus", "project_bonus_eam_fit.molab.py", "Keep both supplied Cu XYZ data files beside the notebook. It fits the model, checks a force, and saves cu_fitted_eam.npz here. Insert those tables into your own edited simulation notebook for the bonus, then check forces and a short trajectory. Compare with the original Mishin model using matched settings."),
        ):
            text = ("Install uv: https://docs.astral.sh/uv/getting-started/installation/\n\n"
                    "Extract the complete project ZIP. Open a terminal in this folder and run:\n\n"
                    f"  uvx marimo@0.25.2 edit --sandbox {notebook}\n\n"
                    "This is an editable marimo notebook. Keep the terminal open while using it. "
                    "The first launch installs Python and the declared packages and needs internet access. "
                    "Later launches reuse the downloaded environment.\n\n" + description + "\n")
            archive.writestr(f"{folder}/START-HERE.txt", text)
        archive.writestr("7-report-template-source.zip", report_sources())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert archive.namelist()[:3] == ["1-assignment.pdf", "2-assignment-tutorial.pdf", "3-assignment-sample-report.pdf"]
        print(f"{output}: {output.stat().st_size:,} bytes")
        for member in archive.infolist():
            print(f"{member.file_size:>10,}  {member.filename}")


if __name__ == "__main__":
    main()

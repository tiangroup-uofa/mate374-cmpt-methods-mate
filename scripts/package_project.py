#!/usr/bin/env python3
"""Build the private student release from the current rendered PDFs and notebook sources.

Render instructor-private/project-demo/{assignment,tutorial}.qmd and the sample report first.
Student names remain in ignored instructor-private/. No study outputs or credentials are included.
"""
import argparse
import hashlib
import io
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "instructor-private/project-demo"
NOTEBOOKS = (
    ("4-md-run", "project_cu_np_melting.molab.py", "Simulation",
     "https://molab.marimo.io/notebooks/nb_snapzMa8oVN3beQpXGuBNc",
     "Keep Cu_mishin1.eam.alloy beside this notebook. The cycle first equilibrates for "
     "100 ps, then continues from the equilibrated positions and velocities. Its main "
     "CSV and XYZ exclude equilibration and use a fresh cycle clock. The initial XYZ "
     "frame is the equilibrated particle. Equilibration energies are saved separately "
     "in *_equilibration.csv. Download the files you want to keep from runs/."),
    ("5-analysis", "project_cu_np_analysis.molab.py", "Analysis",
     "https://molab.marimo.io/notebooks/nb_2m8VYhDJC1A8y2MXkFXtUk",
     "Upload one energy CSV and one atomic-structure XYZ in their separate upload boxes. "
     "For a heating/cooling plot, use the main cycle CSV, not the separate equilibration "
     "CSV. Record melting measurements in Table 3 and complete fit_melting_data with "
     "scipy.optimize.minimize. That incomplete function is the student exercise."),
    ("6-bonus", "project_bonus_eam_fit.molab.py", "Bonus",
     "https://molab.marimo.io/notebooks/nb_6yiNfKPqBZQSU8LNjCPHco",
     "Keep all four supplied input files beside this notebook: Cu_mishin1.eam.alloy, "
     "cu_vasp_train.xyz, cu_vasp_test.xyz and cu_vasp_atom_reference.json. Fit the "
     "15-coefficient EAM model, enter its coefficients in the custom_system skeleton, "
     "and run the checks. Copy the completed function into the Simulation notebook "
     "and select Custom. The empty coefficient array is intentional. Transfer "
     "the fitted model through the completed function."),
)


def report_sources():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        source = (ROOT / "project/report-template/report.tex").read_text()
        archive.writestr("report.tex", source.replace(
            r"\bibliography{../../references}", r"\bibliography{references}"))
        archive.write(ROOT / "references.bib", "references.bib")
        for name in ("fig_stability.png", "fig_cycle.png", "fig_size.png"):
            archive.write(ROOT / "project/report-template" / name, name)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIVATE / "MATE374-group-project.zip")
    parser.add_argument("--public", action="store_true",
                        help="Build the website ZIP using a separately rendered, roster-free assignment.")
    parser.add_argument("--assignment-pdf", type=Path,
                        help="Roster-free assignment PDF, required with --public.")
    args = parser.parse_args()
    output = args.output.resolve()
    assignment = PRIVATE / "assignment.pdf"
    if args.public:
        if not args.assignment_pdf:
            parser.error("--public requires --assignment-pdf with the roster-free PDF.")
        assignment = args.assignment_pdf.resolve()
        if assignment == (PRIVATE / "assignment.pdf").resolve():
            parser.error("Do not publish the named Canvas assignment PDF.")
        text = subprocess.run(["pdftotext", str(assignment), "-"],
                              check=True, capture_output=True, text=True).stdout
        if "Project groups" in text or any(f"Group {n}:" in text for n in range(1, 6)):
            parser.error("The public assignment must not contain the student group roster.")
        allowed = output == ROOT / "project/MATE374-group-project.zip"
        if not allowed and not output.is_relative_to(ROOT / "instructor-private"):
            parser.error("Public output must be the project website ZIP or a private staging file.")
    elif not output.is_relative_to(ROOT / "instructor-private"):
        parser.error("The named Canvas package must stay inside ignored instructor-private/.")
    files = [
        (assignment, "1-assignment.pdf"),
        (PRIVATE / "tutorial.pdf", "2-assignment-tutorial.pdf"),
        (ROOT / "project/report-template/report.pdf", "3-assignment-sample-report.pdf"),
        (ROOT / "activities/project_cu_np_melting.molab.py", "4-md-run/project_cu_np_melting.molab.py"),
        (ROOT / "activities/Cu_mishin1.eam.alloy", "4-md-run/Cu_mishin1.eam.alloy"),
        (ROOT / "activities/project_cu_np_analysis.molab.py", "5-analysis/project_cu_np_analysis.molab.py"),
        (ROOT / "activities/project_bonus_eam_fit.molab.py", "6-bonus/project_bonus_eam_fit.molab.py"),
        (ROOT / "activities/Cu_mishin1.eam.alloy", "6-bonus/Cu_mishin1.eam.alloy"),
        (ROOT / "project/data/cu_vasp_train.xyz", "6-bonus/cu_vasp_train.xyz"),
        (ROOT / "project/data/cu_vasp_test.xyz", "6-bonus/cu_vasp_test.xyz"),
        (ROOT / "project/data/cu_vasp_atom_reference.json", "6-bonus/cu_vasp_atom_reference.json"),
    ]
    for path, _ in files:
        if not path.is_file():
            raise FileNotFoundError(f"Required release input is missing: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, target in files:
            archive.write(source, target)
        links = "\n".join(f"{title}: {url}" for _, _, title, url, _ in NOTEBOOKS)
        archive.writestr("START-HERE.txt",
            "MAT E 374 GROUP PROJECT\n\n"
            "Read 1-assignment.pdf, then 2-assignment-tutorial.pdf. "
            "3-assignment-sample-report.pdf is the report example.\n\n"
            "ONLINE NOTEBOOKS\n" + links + "\n\n"
            "Open the prepared notebooks, sign in and use Fork for your own editable copies. "
            "Choose a GPU machine for MD. Save notebook edits and download results you want "
            "to retain. The PDFs contain clickable links to all three notebooks.\n\n"
            "Folders 4/5/6 contain editable notebook sources and their required input files. "
            "Each folder's START-HERE.txt gives its online link and optional local command. "
            "7-report-template-source.zip contains the editable sample report.\n")
        for folder, notebook, title, url, description in NOTEBOOKS:
            archive.writestr(f"{folder}/START-HERE.txt",
                f"{title.upper()} NOTEBOOK\n\nOpen in molab:\n{url}\n\n"
                "Sign in and use Fork to work in your own editable copy.\n\n"
                + description + "\n\n"
                "OPTIONAL LOCAL FALLBACK\n"
                "Install uv: https://docs.astral.sh/uv/getting-started/installation/\n"
                "Extract the complete ZIP, open a terminal in this folder, then run:\n\n"
                f"  uvx marimo@0.25.2 edit --sandbox {notebook}\n\n"
                "The first launch resolves the notebook's declared dependencies and needs "
                "internet access. Keep the terminal open and save edits before closing.\n")
        archive.writestr("7-report-template-source.zip", report_sources())
        checksums = "\n".join(
            f"{hashlib.sha256(archive.read(name)).hexdigest()}  {name}"
            for name in archive.namelist())
        archive.writestr("SHA256SUMS.txt", checksums + "\n")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert archive.namelist()[:3] == [
            "1-assignment.pdf", "2-assignment-tutorial.pdf", "3-assignment-sample-report.pdf"]
        assert len(set(archive.namelist())) == len(archive.namelist())
        print(f"{output}: {output.stat().st_size:,} bytes")
        for member in archive.infolist():
            print(f"{member.file_size:>10,}  {member.filename}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


SOURCE_DIR = "activities"
OUTPUT_DIR = "wasm-local"
MANIFEST_NAME = ".marimo-export-manifest.json"
AUTO_RUN_OPT_OUT = "mate374: auto-run = false"
BUILD_EXECUTION_OPT_OUT = "mate374: build-execute = false"
NON_RENDERED_QMD_SOURCES = {"assignments/A2/q3-lj-estimation-backup.qmd"}
FULL_SITE_QMD_DIRS = (
    "syllabus",
    "introduction",
    "units",
    "assignments",
    "project",
    "seminars",
)
LOCAL_WASM_DIV_RE = re.compile(
    r"(?m)^\s*:{3,}\s*\{[^}]*\.quarto-wasm-local\b[^}]*\}"
)
NOTEBOOK_ATTRIBUTE_RE = re.compile(r"\b(?:notebook|source)\s*=\s*([\"'])(.*?)\1")
LOCAL_WASM_HTML_RE = re.compile(r"wasm-local/([^/\s)\"'?#]+)\.html")


def contains_import_marimo(py_path: Path) -> bool:
    """Only export Python files that look like marimo notebooks."""
    try:
        return "marimo" in py_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def export_mode_and_outstem(py_file: Path) -> tuple[str, str]:
    """Map ``foo.py`` to run mode and ``foo.edit.py`` to editable mode."""
    stem = py_file.stem
    if stem.endswith(".edit"):
        return "edit", stem.removesuffix(".edit")
    return "run", stem


def rendered_qmd_sources(project_root: Path) -> list[Path]:
    """Mirror the full-profile page roots, excluding the intentionally omitted unit."""
    pages = set(project_root.glob("*.qmd"))
    for directory in FULL_SITE_QMD_DIRS:
        source_dir = project_root / directory
        if source_dir.is_dir():
            pages.update(source_dir.rglob("*.qmd"))

    return sorted(
        path
        for path in pages
        if path.is_file()
        and not path.name.startswith(".#")
        and path.relative_to(project_root).as_posix() not in NON_RENDERED_QMD_SOURCES
        and path.relative_to(project_root).as_posix() != "units/03"
        and not path.relative_to(project_root).as_posix().startswith("units/03/")
    )


def page_export_requirements(
    project_root: Path, source_dir: Path
) -> tuple[set[Path], set[str]]:
    """Collect embedded notebooks and direct links to generated WASM HTML."""
    notebook_sources: set[Path] = set()
    linked_html_stems: set[str] = set()

    for qmd_path in rendered_qmd_sources(project_root):
        text = qmd_path.read_text(encoding="utf-8", errors="ignore")
        for div_match in LOCAL_WASM_DIV_RE.finditer(text):
            attributes = div_match.group(0)
            notebook_match = NOTEBOOK_ATTRIBUTE_RE.search(attributes)
            if notebook_match is None:
                raise ValueError(
                    f"A .quarto-wasm-local Div in {qmd_path} needs a notebook or source attribute"
                )

            notebook = notebook_match.group(2)
            notebook_path = Path(notebook)
            if (
                notebook_path.is_absolute()
                or ".." in notebook_path.parts
                or "\\" in notebook
                or not notebook.endswith(".py")
            ):
                raise ValueError(
                    f"Invalid local WASM notebook path in {qmd_path}: {notebook}"
                )

            source_path = (
                project_root / notebook_path
                if "/" in notebook
                else source_dir / notebook_path
            )
            notebook_sources.add(source_path.resolve())

        linked_html_stems.update(LOCAL_WASM_HTML_RE.findall(text))

    return notebook_sources, linked_html_stems


def export_fingerprint(project_root: Path, notebooks: list[Path]) -> str:
    """Hash every input that can change the generated WASM bundle."""
    digest = hashlib.sha256()
    digest.update(b"marimo-html-wasm-v3\0--execute\0auto-run\0")
    inputs = [
        Path(__file__).resolve(),
        project_root / "pyproject.toml",
        project_root / "uv.lock",
    ]
    inputs.extend(notebooks)
    for path in inputs:
        digest.update(str(path.relative_to(project_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def cache_is_current(
    target_dir: Path, notebooks: list[Path], fingerprint: str
) -> bool:
    manifest_path = target_dir / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    expected_html = {
        f"{export_mode_and_outstem(notebook)[1]}.html" for notebook in notebooks
    }
    return (
        manifest.get("fingerprint") == fingerprint
        and set(manifest.get("html", [])) == expected_html
        and all((target_dir / filename).is_file() for filename in expected_html)
        and (target_dir / "assets").is_dir()
    )


def enable_browser_auto_run(output_html: Path, notebook: Path) -> None:
    """Make trusted course exports live immediately in editable WASM mode."""
    if AUTO_RUN_OPT_OUT in notebook.read_text(encoding="utf-8", errors="ignore"):
        return

    html = output_html.read_text(encoding="utf-8")
    html, replacements = re.subn(
        r'("auto_instantiate"\s*:\s*)false',
        r"\1true",
        html,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError(
            "Could not enable browser auto-run in marimo export: "
            f"{output_html}"
        )
    output_html.write_text(html, encoding="utf-8")


def expose_browser_save(output_html: Path) -> None:
    """Show marimo's existing save button in editable WASM exports.

    marimo hides this button in standalone WASM HTML because there is no
    server-side file to save. The browser save path still works, however:
    Cmd/Ctrl-S downloads the current notebook and updates browser recovery
    state. Showing the button gives students the same path visibly.
    """
    html = output_html.read_text(encoding="utf-8")
    html, replacements = re.subn(
        r"\n\s*#save-button\s*\{\s*display:\s*none\s*!important;\s*\}",
        "",
        html,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError(
            "Could not expose browser save button in marimo export: "
            f"{output_html}"
        )
    output_html.write_text(html, encoding="utf-8")


def export_all(
    project_root: Path,
    notebooks: list[Path],
    target_dir: Path,
    fingerprint: str,
) -> None:
    temporary_dir = Path(tempfile.mkdtemp(prefix="marimo_export_"))
    try:
        for notebook in notebooks:
            mode, output_stem = export_mode_and_outstem(notebook)
            output_html = temporary_dir / f"{output_stem}.html"
            source = notebook.read_text(encoding="utf-8", errors="ignore")
            execute_flag = (
                "--no-execute"
                if BUILD_EXECUTION_OPT_OUT in source
                else "--execute"
            )
            command = [
                "uv",
                "run",
                "--locked",
                "marimo",
                "export",
                "html-wasm",
                "--mode",
                mode,
                "--force",
                execute_flag,
            ]
            if mode == "run":
                command.append("--no-show-code")
            command.extend([str(notebook), "-o", str(output_html)])

            print(f"Run {' '.join(command)}...")
            subprocess.run(
                command,
                check=True,
                cwd=project_root,
                stdin=subprocess.DEVNULL,
            )
            if mode == "edit":
                enable_browser_auto_run(output_html, notebook)
                expose_browser_save(output_html)
            shutil.copy2(notebook, temporary_dir)

        html_files = sorted(
            f"{export_mode_and_outstem(notebook)[1]}.html" for notebook in notebooks
        )
        (temporary_dir / MANIFEST_NAME).write_text(
            json.dumps({"fingerprint": fingerprint, "html": html_files}, indent=2)
            + "\n",
            encoding="utf-8",
        )

        # Every notebook is exported into one directory, so all generated HTML
        # files reuse the same hashed marimo assets.
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(temporary_dir), str(target_dir))
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def main() -> None:
    project_root = Path.cwd()
    source_dir = project_root / SOURCE_DIR

    if not source_dir.is_dir():
        raise SystemExit(f"Expected {SOURCE_DIR!r} folder at project root: {source_dir}")

    # Scan only direct activities/ children: activities/graveyard is retained
    # for recovery but is deliberately outside the export source set.
    candidates = sorted(
        path
        for path in [*source_dir.iterdir(), *(project_root / "assignments").rglob("*.py")]
        if path.is_file()
        and path.suffix == ".py"
        and not path.name.endswith(".molab.py")
        and contains_import_marimo(path)
    )
    referenced_paths, linked_html_stems = page_export_requirements(
        project_root, source_dir
    )
    candidates_by_path = {path.resolve(): path for path in candidates}
    missing_sources = referenced_paths - candidates_by_path.keys()
    if missing_sources:
        missing = ", ".join(sorted(str(path) for path in missing_sources))
        raise SystemExit(f"Current Quarto pages reference unavailable Marimo notebooks: {missing}")

    candidates_by_stem: dict[str, list[Path]] = {}
    for path in candidates:
        stem = export_mode_and_outstem(path)[1]
        candidates_by_stem.setdefault(stem, []).append(path)
    missing_html = linked_html_stems - candidates_by_stem.keys()
    if missing_html:
        raise SystemExit(
            "Current Quarto pages link to WASM HTML without a source notebook: "
            + ", ".join(sorted(missing_html))
        )

    notebooks = sorted(
        {
            *(candidates_by_path[path] for path in referenced_paths),
            *(path for stem in linked_html_stems for path in candidates_by_stem[stem]),
        }
    )
    output_stems = [export_mode_and_outstem(path)[1] for path in notebooks]
    if len(output_stems) != len(set(output_stems)):
        raise SystemExit("Marimo notebook filenames map to duplicate HTML output names")

    # Export into a generated source resource. Quarto copies this directory to
    # the output site after pre-render, so preview knows how to serve every
    # nested CSS, font, worker, and JavaScript asset.
    target_dir = project_root / OUTPUT_DIR
    fingerprint = export_fingerprint(project_root, notebooks)

    # Quarto preview may invoke pre-render more than once while serving pages.
    # Serialize exporters, then recheck the manifest after acquiring the lock.
    lock_path = project_root / ".quarto" / "marimo-export.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if cache_is_current(target_dir, notebooks, fingerprint):
            print("Marimo WASM exports are current; skipping export.")
            return
        export_all(project_root, notebooks, target_dir, fingerprint)


if __name__ == "__main__":
    main()

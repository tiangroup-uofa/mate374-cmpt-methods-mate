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
                "--execute",
            ]
            if mode == "run":
                command.append("--no-show-code")
            command.extend([str(notebook), "-o", str(output_html)])

            print(f"Run {' '.join(command)}...")
            subprocess.run(command, check=True, cwd=project_root)
            enable_browser_auto_run(output_html, notebook)
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

    notebooks = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix == ".py"
        and not path.name.endswith(".molab.py")
        and contains_import_marimo(path)
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

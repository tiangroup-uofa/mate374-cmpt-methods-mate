#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SOURCE_DIR = "activities"
OUTPUT_DIR = "wasm-local"
MANIFEST_NAME = ".marimo-export-manifest.json"
WASM_MANIFEST_VERSION = 2
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
    """Scan the course page roots for notebook references, including every unit."""
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
    )


def input_qmd_sources(project_root: Path) -> list[Path] | None:
    """Return the pages in the current Quarto render, or None outside Quarto.

    Quarto passes the files it is about to render to pre-render scripts. A
    preview re-render of one edited page therefore exports only the notebooks
    that page embeds or links.
    """
    listing_path = os.environ.get("QUARTO_USE_FILE_FOR_PROJECT_INPUT_FILES")
    if listing_path:
        listing = Path(listing_path).read_text(encoding="utf-8")
    else:
        listing = os.environ.get("QUARTO_PROJECT_INPUT_FILES")
    if listing is None:
        return None

    site_pages = set(rendered_qmd_sources(project_root))
    pages = {(project_root / line.strip()).resolve() for line in listing.splitlines()}
    return sorted(page for page in pages if page in site_pages)


def page_notebook_references(
    qmd_path: Path, project_root: Path, source_dir: Path
) -> tuple[set[Path], set[str]]:
    """Collect one page's embedded notebooks and direct links to WASM HTML."""
    notebook_sources: set[Path] = set()
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

    return notebook_sources, set(LOCAL_WASM_HTML_RE.findall(text))


def page_export_requirements(
    project_root: Path, source_dir: Path, pages: list[Path] | None = None
) -> tuple[set[Path], set[str]]:
    """Collect embedded notebooks and direct WASM HTML links for the pages."""
    notebook_sources: set[Path] = set()
    linked_html_stems: set[str] = set()
    if pages is None:
        pages = rendered_qmd_sources(project_root)
    for qmd_path in pages:
        sources, stems = page_notebook_references(qmd_path, project_root, source_dir)
        notebook_sources.update(sources)
        linked_html_stems.update(stems)
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


def export_notebook(project_root: Path, notebook: Path, output_dir: Path) -> Path:
    """Export one notebook, plus marimo's shared assets, into output_dir."""
    mode, output_stem = export_mode_and_outstem(notebook)
    output_html = output_dir / f"{output_stem}.html"
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
        # The locked project environment already has every notebook
        # dependency. marimo's per-notebook sandbox would instead resolve the
        # newest marimo release, which costs time and ignores uv.lock.
        "--no-sandbox",
        execute_flag,
    ]
    if mode == "run":
        command.append("--no-show-code")
    command.extend([str(notebook), "-o", str(output_html)])

    # Exports run concurrently, so print each notebook's log in one block.
    result = subprocess.run(
        command,
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(f"Run {' '.join(command)}", result.stdout, sep="\n", flush=True)
        raise subprocess.CalledProcessError(result.returncode, command)
    warnings = [line for line in result.stdout.splitlines() if "warn" in line.lower()]
    print(f"Exported {notebook.name}", *warnings, sep="\n  ", flush=True)
    if mode == "edit":
        enable_browser_auto_run(output_html, notebook)
        expose_browser_save(output_html)
    shutil.copy2(notebook, output_dir)
    return output_html


def export_jobs() -> int:
    """Number of notebooks exported at once; MATE374_EXPORT_JOBS overrides."""
    configured = os.environ.get("MATE374_EXPORT_JOBS")
    if configured:
        return max(1, int(configured))
    return os.cpu_count() or 1


def export_in_parallel(project_root: Path, notebooks: list[Path], scratch_dir: Path):
    """Export notebooks concurrently and yield each finished (notebook, html).

    Every export writes to its own subdirectory because marimo rewrites the
    assets folder beside its output. Callers merge the hashed assets.
    """
    failed: list[Path] = []
    with ThreadPoolExecutor(max_workers=export_jobs()) as pool:
        futures = {}
        for notebook in notebooks:
            output_dir = scratch_dir / export_mode_and_outstem(notebook)[1]
            output_dir.mkdir(parents=True)
            futures[pool.submit(export_notebook, project_root, notebook, output_dir)] = notebook
        for future in as_completed(futures):
            try:
                output_html = future.result()
            except subprocess.CalledProcessError:
                failed.append(futures[future])
                continue
            yield futures[future], output_html
    if failed:
        raise SystemExit(
            "marimo export failed for: " + ", ".join(str(path) for path in sorted(failed))
        )


def export_all(
    project_root: Path,
    notebooks: list[Path],
    target_dir: Path,
    fingerprint: str,
) -> None:
    temporary_dir = Path(tempfile.mkdtemp(prefix="marimo_export_"))
    try:
        with tempfile.TemporaryDirectory(prefix="marimo_scratch_") as scratch:
            for notebook, output_html in export_in_parallel(
                project_root, notebooks, Path(scratch)
            ):
                merge_assets(output_html.parent / "assets", temporary_dir / "assets")
                shutil.copy2(notebook, temporary_dir / notebook.name)
                shutil.copy2(output_html, temporary_dir / output_html.name)

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


def shared_export_fingerprint(project_root: Path) -> str:
    """Hash the inputs shared by every WASM export.

    The marimo version (uv.lock) and the export code decide the HTML shell and
    the hashed assets. A change here invalidates every cached notebook.
    """
    digest = hashlib.sha256()
    digest.update(f"marimo-html-wasm-v{WASM_MANIFEST_VERSION}\0".encode())
    for function in (export_notebook, enable_browser_auto_run, expose_browser_save):
        digest.update(inspect.getsource(function).encode())
        digest.update(b"\0")
    for path in (project_root / "pyproject.toml", project_root / "uv.lock"):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def notebook_fingerprint(project_root: Path, notebook: Path) -> str:
    digest = hashlib.sha256()
    digest.update(notebook.relative_to(project_root).as_posix().encode())
    digest.update(b"\0")
    digest.update(notebook.read_bytes())
    return digest.hexdigest()


def read_wasm_manifest(target_dir: Path, shared: str) -> dict:
    try:
        manifest = json.loads((target_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    if (
        manifest.get("version") == WASM_MANIFEST_VERSION
        and manifest.get("shared") == shared
        and isinstance(manifest.get("notebooks"), dict)
        and (target_dir / "assets").is_dir()
    ):
        return manifest

    # New marimo version, export code, or manifest layout: start clean so
    # superseded hashed assets do not accumulate.
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    return {"version": WASM_MANIFEST_VERSION, "shared": shared, "notebooks": {}}


def write_wasm_manifest(target_dir: Path, manifest: dict) -> None:
    manifest_path = target_dir / MANIFEST_NAME
    partial_path = manifest_path.with_suffix(".tmp")
    partial_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial_path, manifest_path)


def prune_wasm_exports(target_dir: Path, manifest: dict, keep_html: set[str]) -> None:
    """Remove exports for notebooks that no course page references anymore."""
    entries = manifest["notebooks"]
    for html_name in sorted(set(entries) - keep_html):
        print(f"Remove unreferenced marimo export {html_name}")
        del entries[html_name]
    tracked = {MANIFEST_NAME, "assets"}
    for html_name, entry in entries.items():
        tracked.update({html_name, Path(entry["source"]).name})
    for path in target_dir.iterdir():
        if path.name in tracked:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def merge_assets(source_assets: Path, target_assets: Path) -> None:
    """Copy marimo assets the bundle does not have yet.

    Asset filenames carry content hashes, so an existing name already holds
    the same file.
    """
    for path in source_assets.rglob("*"):
        if path.is_dir():
            continue
        destination = target_assets / path.relative_to(source_assets)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def export_stale_notebooks(
    project_root: Path,
    notebooks: list[Path],
    target_dir: Path,
    keep_html: set[str],
) -> None:
    """Export only notebooks whose source changed since their cached export."""
    manifest = read_wasm_manifest(target_dir, shared_export_fingerprint(project_root))
    prune_wasm_exports(target_dir, manifest, keep_html)
    write_wasm_manifest(target_dir, manifest)

    entries = manifest["notebooks"]
    stale: list[tuple[Path, str]] = []
    for notebook in notebooks:
        html_name = f"{export_mode_and_outstem(notebook)[1]}.html"
        fingerprint = notebook_fingerprint(project_root, notebook)
        entry = entries.get(html_name, {})
        if entry.get("fingerprint") != fingerprint or not (target_dir / html_name).is_file():
            stale.append((notebook, fingerprint))

    if not stale:
        print(f"Marimo WASM exports are current for {len(notebooks)} notebook(s).")
        return
    print(
        f"Exporting {len(stale)} of {len(notebooks)} marimo notebook(s) to "
        f"{OUTPUT_DIR}/ with {min(export_jobs(), len(stale))} parallel job(s).",
        flush=True,
    )

    fingerprints = dict(stale)
    with tempfile.TemporaryDirectory(prefix="marimo_export_") as temporary:
        for notebook, output_html in export_in_parallel(
            project_root, list(fingerprints), Path(temporary)
        ):
            fingerprint = fingerprints[notebook]
            # Install assets before the HTML that loads them, and record each
            # finished notebook so a later failure keeps earlier work.
            merge_assets(output_html.parent / "assets", target_dir / "assets")
            shutil.copy2(notebook, target_dir / notebook.name)
            shutil.copy2(output_html, target_dir / output_html.name)
            entries[output_html.name] = {
                "fingerprint": fingerprint,
                "source": notebook.relative_to(project_root).as_posix(),
            }
            write_wasm_manifest(target_dir, manifest)


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
    # Validate and export what the pages in this render need. A whole-site
    # render passes every page; a preview re-render passes the edited page.
    referenced_paths, linked_html_stems = page_export_requirements(
        project_root, source_dir, input_qmd_sources(project_root)
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

    # Keep cached exports for every notebook any course page still uses, even
    # when this render covers only some pages.
    site_paths, site_html_stems = set(), set()
    for page in rendered_qmd_sources(project_root):
        try:
            sources, stems = page_notebook_references(page, project_root, source_dir)
        except ValueError:
            continue
        site_paths.update(sources)
        site_html_stems.update(stems)
    keep_html = {f"{export_mode_and_outstem(path)[1]}.html" for path in site_paths}
    keep_html.update(f"{stem}.html" for stem in site_html_stems)

    # Export into a generated source resource. Quarto copies this directory to
    # the output site after pre-render, so preview knows how to serve every
    # nested CSS, font, worker, and JavaScript asset.
    target_dir = project_root / OUTPUT_DIR

    # Quarto preview may invoke pre-render more than once while serving pages.
    # Serialize exporters, then recheck the manifest after acquiring the lock.
    lock_path = project_root / ".quarto" / "marimo-export.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        export_stale_notebooks(project_root, notebooks, target_dir, keep_html)


if __name__ == "__main__":
    main()

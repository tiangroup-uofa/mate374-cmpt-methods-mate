# MATE 374 — Computational Methods in Materials Engineering

Quarto source for the redesigned University of Alberta MATE 374 course. Quarto owns the course narrative and printable output; marimo provides browser-side computational experiments.

## Requirements

- [Quarto](https://quarto.org/) 1.9 or newer
- [uv](https://docs.astral.sh/uv/)
- Python 3.11 or newer

## Preview or render

For everyday writing, preview HTML or build the HTML site without compiling PDFs:

```bash
quarto preview
quarto render --to html
```

For a complete local check, use the same command as CI:

```bash
quarto render --profile full
```

Both paths export notebooks from `activities/` into the ignored `wasm-local/` resource directory. Unchanged exports are reused; Quarto copies the complete bundle into the built site.

The `full` profile also regenerates scripted course figures, builds and stages answer PDFs and enables the HTML pages' PDF download links. Plain preview and HTML-only builds skip the answer-PDF hook and hide those format links. Explicit handout links in the page text may still point to PDFs from an earlier full build; run the full build to refresh them.

Keep both output-format definitions in `_quarto.yml` so page-specific PDF settings remain available. A bare `quarto render` renders both formats but omits the answer-PDF hook; use `--to html` for daily builds and `--profile full` for complete builds.

## Deployment

Pushes to `main` run [`.github/workflows/render_pages.yml`](.github/workflows/render_pages.yml). The workflow installs the locked `uv` environment, Quarto, and TinyTeX, then runs `quarto render --profile full` to export notebooks, regenerate scripted figures, stage answer PDFs, render all HTML and PDF pages, and deploy `_site/` with GitHub Pages. The post-render hook generates the agent-facing `llms.txt` index.

Figure generation is managed by [`scripts/render_figures.py`](scripts/render_figures.py), which runs the plotting scripts from source and saved data without rerunning benchmarks. Add new figure generators to its explicit list. To refresh figures without a full site build, run `uv run --locked python scripts/render_figures.py`. Daily HTML builds reuse the existing images.

The workflow follows the CHE 318 and MATE 664 Pages pattern, with `uv sync --locked` added so local and CI Python environments use the same dependency resolution. The full build also needs `librsvg2-bin` for SVG-to-PDF image conversion. CI still produces the complete site on every push; the HTML-only path speeds up local iteration.

## Course identity

The site uses University of Alberta evergreen and gold, the university crest as its favicon, and the official UAlberta and Faculty of Engineering marks in the home-page banner. The navbar remains text-only for a clean, reliable layout. See [`assets/BRANDING.md`](assets/BRANDING.md) for asset provenance and the corresponding UAlberta CSS pattern.

## Reading and class views

Lecture pages use one rendered HTML document for both student reading and classroom display. Add `body-classes: lecture-page` to a lecture's YAML. The injected view switch changes the URL in place:

```text
.../L01/             # reading view
.../L01/?view=class  # full-width, larger classroom view
```

A single bottom-right icon toggles the two states while preserving the page, scroll position, and interactive notebook sessions; it does not maintain a duplicate slide deck. [`assets/class-view.html`](assets/class-view.html) manages URL state and [`styles.css`](styles.css) defines the two layouts.

## Orientation lectures and marimo workflow

[`units/01/L01/index.qmd`](units/01/L01/index.qmd) introduces the question–model–calculation workflow and gives students a small thermal-expansion function to edit. [`units/01/L02/index.qmd`](units/01/L02/index.qmd) turns a π series into an algorithm and compares loops, arrays, timing, error, and floating-point representation.

The earlier iframe and multi-playground deployment experiments are retained under [`units/archive/initial-prototype/00-orientation/archive/`](units/archive/initial-prototype/00-orientation/archive/index.qmd).

### Hosted source iframe

```markdown
::: {.marimo-iframe source="activities/example.edit.py" height="700"}
Static fallback for non-HTML output.
:::
```

[`filters/marimo-iframe.lua`](filters/marimo-iframe.lua) calls [`scripts/marimo_iframe_url.py`](scripts/marimo_iframe_url.py) to compress the notebook into a `marimo.app` source URL.

### Locally exported WASM iframe

```markdown
::: {.quarto-wasm-local notebook="example.edit.py" height="700"}
Static fallback for non-HTML output.
:::
```

For a dashboard-style app that benefits from the whole display, opt in to the course fullscreen control without restoring marimo's full chrome:

```markdown
::: {.quarto-wasm-local notebook="dashboard.py" height="700" fullscreen="true"}
Static fallback for non-HTML output.
:::
```

`loading="eager"` starts an iframe immediately; the default is `loading="lazy"`. Run-mode `*.py` dashboards expose only requested controls. Editable `*.edit.py` notebooks also receive the **App view / Edit code** control. Every local embed includes a collapsed **Notebook not loading?** panel with direct-open and reload/reset actions.

[`_marimo_export.py`](_marimo_export.py), adapted from MATE 664, exports all notebooks into the generated `wasm-local/` resource directory; Quarto copies it to `_site/wasm-local/`. This pre-render staging makes Quarto preview register every nested CSS, font, worker, and JavaScript asset. The notebooks share one generated marimo `assets/` directory. [`filters/quarto-wasm-local.lua`](filters/quarto-wasm-local.lua) maps the notebook filename to its local HTML output and supplies the iframe attributes.

Filename convention:

- `example.py` → run-mode `example.html`
- `example.edit.py` → editable `example.html`
- `example.molab.py` → excluded from WASM export and launched in molab

### Notebook startup policy

Course notebooks run at browser startup by default. `--execute` provides an immediate build-time preview, while the trusted course exporter enables marimo's browser `auto_instantiate` setting so the live kernel runs the complete dependency graph as soon as it is ready. Refreshing therefore resets and reruns the authored notebook without asking students to click **Run all**.

For an exceptional expensive notebook, add this source marker and provide an explicit `mo.ui.run_button` inside the notebook:

```python
# mate374: auto-run = false
```

Native activities that should not start in the browser belong in molab instead.

For non-HTML formats, each filter emits the contents of its fenced Div as the static fallback.

### Cloud notebooks in molab

Notebooks that require a native cloud environment—such as a future `mace-torch` activity—should launch in molab instead of pretending to be WASM-compatible. Once the notebook is committed to GitHub, use a normal molab link without the `/wasm` suffix:

```markdown
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/tiangroup-uofa/mate374-cmpt-methods-mate/blob/main/activities/numba_pi.molab.py)
```

Keep a concise Quarto explanation and static result beside the link so the course page remains readable without starting the cloud runtime.

The canonical test notebook is [`activities/inline_numpy.edit.py`](activities/inline_numpy.edit.py).

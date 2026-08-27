# MATE 374 — Computational Methods in Materials Engineering

Quarto source for the redesigned University of Alberta MATE 374 course. Quarto owns the course narrative and printable output; marimo provides browser-side computational experiments.

## Requirements

- [Quarto](https://quarto.org/) 1.9 or newer
- [uv](https://docs.astral.sh/uv/)
- Python 3.11 or newer

## Preview or render

```bash
quarto preview
# or
quarto render
```

The pre-render hook exports notebooks from `activities/` into the ignored `wasm-local/` resource directory. Quarto then copies the complete bundle into the built site.

## Continuous deployment

[`.github/workflows/pages.yml`](.github/workflows/pages.yml) renders the site with `quarto render` (using `uv` for the pinned Python environment, matching the pre-render marimo export and iframe filters above) on every push to `main` and publishes the result to GitHub Pages. Pull requests trigger the same render as a build check without deploying. The repository's **Settings → Pages** source must be set to "GitHub Actions" for the deploy job to publish.

Because `_quarto.yml` declares both `html` and `pdf` output formats, `quarto render` builds a PDF for every page as well as the HTML site, so the workflow installs TinyTeX (`quarto-dev/quarto-actions/setup` with `tinytex: true`) and `librsvg2-bin` (for SVG-to-PDF image conversion in the LaTeX build).

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

[`units/00-orientation/L01/index.qmd`](units/00-orientation/L01/index.qmd) introduces the question–model–calculation workflow and gives students a small thermal-expansion function to edit. [`units/00-orientation/L02/index.qmd`](units/00-orientation/L02/index.qmd) turns a π series into an algorithm and compares loops, arrays, timing, error, and floating-point representation.

The earlier iframe and multi-playground deployment experiments are retained under [`units/00-orientation/archive/`](units/00-orientation/archive/index.qmd).

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

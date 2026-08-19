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

The post-render hook exports notebooks from `activities/` after Quarto builds the site.

## Reading and class views

Lecture pages use one rendered HTML document for both student reading and classroom display. Add `body-classes: lecture-page` to a lecture's YAML. The injected view switch changes the URL in place:

```text
.../L01/             # reading view
.../L01/?view=class  # full-width, larger classroom view
```

The switch preserves the page, scroll state, and interactive notebook sessions; it does not maintain a duplicate slide deck. [`assets/class-view.html`](assets/class-view.html) manages URL state and [`styles.css`](styles.css) defines the two layouts.

## Marimo embedding comparison

[`units/00-orientation/L01/index.qmd`](units/00-orientation/L01/index.qmd) renders the same editable notebook using two iframe sources.

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

[`_marimo_export.py`](_marimo_export.py), adapted from MATE 664, exports all notebooks into `_site/wasm-local/`. They share one generated marimo `assets/` directory. [`filters/quarto-wasm-local.lua`](filters/quarto-wasm-local.lua) maps the notebook filename to its local HTML output and supplies the iframe attributes.

Filename convention:

- `example.py` → run-mode `example.html`
- `example.edit.py` → editable `example.html`

For non-HTML formats, each filter emits the contents of its fenced Div as the static fallback.

The canonical test notebook is [`activities/inline_numpy.edit.py`](activities/inline_numpy.edit.py).

# Lecture 6 and 7 baseline

Snapshot of the working files before the focused L06/L07 round. This preserves the user's uncommitted drafts and configuration, rather than the Git HEAD versions.

Contents:
- `units/02/L06/` and `units/02/L07/`: drafts and existing undo-history files.
- `assets/L06-bracket-method-missed-roots.{png,svg}`: associated figures.
- `_quarto.yml`: original configuration before limiting rendering and disabling build hooks.
- `SHA256.json`: checksums verified against the originals when copied.

From the project root, restore the full rendering configuration:

```sh
cp _archive/lectures-06-07-20260915-160126/_quarto.yml _quarto.yml
```

To restore the lecture drafts (overwrites subsequent edits):

```sh
cp _archive/lectures-06-07-20260915-160126/units/02/L06/index.qmd units/02/L06/index.qmd
cp _archive/lectures-06-07-20260915-160126/units/02/L07/index.qmd units/02/L07/index.qmd
```

Working instructions are recorded in the root `AGENTS.md`. No render or WASM build was needed for this setup.

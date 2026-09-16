# Project instructions

## Current work: Lectures 6 and 7

- Prioritize completing both lecture drafts over polishing demos. Update lecture names on landing pages to match, but preserve the Unit 02 title **Working with Functions**. Do not rename the unit.
- Focus this round on `units/02/L06/index.qmd` and `units/02/L07/index.qmd`. Preserve the drafts' structure, voice, and mathematical visuals; the user is proud of them.
- Put concepts and physical interpretation first. Keep heavy mathematics and derivations in collapsed, expandable sections. Check for additional passages that need folding.
- For the current L06 refinements, the user has authorized direct edits to the local WASM notebook sources in `activities/`. Keep `student_map` in its own visible cell; collapse the other fixed-point demo code cells. If pairing on a live molab session later, use `marimo-pair` and edit through `marimo._code_mode`.
- Keep iteration short. Check the L06 notebooks and their numerical examples locally; leave full-site rendering to CI. Avoid long local Quarto renders or WASM/browser debugging.
- Use **Newton–Raphson** consistently in teaching prose and demo labels. Preserve required API names such as SciPy's `method="newton"`.
- Full-site rendering is restored in `_quarto.yml` at the user's request. Pre-render hooks export all local notebooks and stage answer PDFs; the post-render hook generates `llms.txt`. Push completed changes to GitHub for CI rendering rather than running a full local build.
- Original drafts, associated L06 figures, and the pre-focus `_quarto.yml` are saved in `_archive/lectures-06-07-20260915-160126/`, with SHA-256 checksums. The current full `_quarto.yml` was additionally checkpointed at `_archive/quarto-checkpoints/20260916-102601-before-l06-preview/`. Preserve these baselines.
- The archived `_quarto.yml` records the baseline rendering configuration. Preserve the current navigation, including S03, when changing build hooks.
- Molab session URL: `https://sb-93d3852d0dfbeff0.sb.molab.run/`. Use the token supplied in the conversation; do not save it in project files. Request a fresh connection if the session expires.

## Writing style

When the user asks for writing refinement, use two passes. The first pass may be drafted freely, but the second pass must proofread every sentence and remove garbage wording.

Avoid filler and defensive contrast patterns such as “this is X, not Y” or “this demo is only for X, not a real Y.” State the useful point directly. Preserve the user’s intended meaning, tone, and edits rather than replacing them with generic polished prose.

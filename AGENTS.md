# Project instructions

## Current work: Lectures 6 and 7

- Prioritize completing both lecture drafts over polishing demos. Update lecture names on landing pages to match, but preserve the Unit 02 title **Working with Functions**. Do not rename the unit.
- Focus this round on `units/02/L06/index.qmd` and `units/02/L07/index.qmd`. Preserve the drafts' structure, voice, and mathematical visuals; the user is proud of them.
- Put concepts and physical interpretation first. Keep heavy mathematics and derivations in collapsed, expandable sections. Check for additional passages that need folding.
- Develop and test demos interactively in the user-provided molab session using the `marimo-pair` skill. Edit live notebook cells through `marimo._code_mode`, not the running notebook's file.
- Keep iteration short. Do not block progress on long Quarto renders or local WASM/browser debugging. Validate in molab first, then keep the local notebook/export reproducible for later WASM use.
- `_quarto.yml` now includes the full-site render targets, including seminars. Whole-project pre/post-render hooks remain disabled. Do not re-enable them without the user's agreement.
- Original drafts, associated L06 figures, and the pre-focus `_quarto.yml` are saved in `_archive/lectures-06-07-20260915-160126/`, with SHA-256 checksums. Preserve this baseline.
- The archived `_quarto.yml` records the original render hooks. Restore hooks selectively if agreed; preserve the current navigation, including S03.
- Molab session URL: `https://sb-93d3852d0dfbeff0.sb.molab.run/`. Use the token supplied in the conversation; do not save it in project files. Request a fresh connection if the session expires.

## Writing style

When the user asks for writing refinement, use two passes. The first pass may be drafted freely, but the second pass must proofread every sentence and remove garbage wording.

Avoid filler and defensive contrast patterns such as “this is X, not Y” or “this demo is only for X, not a real Y.” State the useful point directly. Preserve the user’s intended meaning, tone, and edits rather than replacing them with generic polished prose.

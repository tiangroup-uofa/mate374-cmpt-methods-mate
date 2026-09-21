# Project instructions

## Figure-generation policy

- All script-generated figures must use `dpi=300` when saved. Do not introduce lower-resolution figure output unless the instructor explicitly requests it.

## Current work: Lectures 8 and 9

- Follow the user's L08/L09 planning comments and preserve the tone and lecture structure established in L01–L07. Cover interpolation and regression in these two lectures; prioritize concepts and practical Python calls over exhaustive derivations.
- The original commented plans are saved, with checksums, in `_archive/lectures-08-09-20260920-180708-planning/`. The book extract is `units/02/l08-09-kiusalaas-book-extract.txt`. Next-round questions and the teaching route are in `units/02/l08-l09-review.md`.
- The instructor approved restoring full-site rendering and pushing after the final L08 comments. `_quarto.yml` now uses the full render targets and shared notebook exporter. Use `quarto preview` or `quarto render --to html` for daily work, and `quarto render --profile full` for complete local/CI builds.
- Preserve the L09 and S03 navigation. The full pre-focus configuration is checkpointed at `_archive/quarto-checkpoints/20260920-182224-before-l08-l09-focus/`. The focused exporter remains available in `scripts/export_l08_l09.py` for targeted checks, but is no longer a default build hook.
- Keep notebook prose light, with editable numerical calls visible and plotting/support code collapsed. Check examples with `uv run --locked python scripts/l08_l09_figures.py --check-only` and `marimo check`.
- In L08/L09, show how the conditions become systems of equations. Label known data, chosen function forms, and unknown coefficients explicitly. Introduce unfamiliar notation through expanded expressions and concrete examples. Keep the Newton divided-difference definitions and procedure table visible. Concision must not remove the mathematics students need to read the method.
- Preserve the Unit 02 title **Working with Functions**. Leave four major refinement questions for the instructor rather than blocking the first draft on decisions.

## Earlier work: Lectures 6 and 7

- Prioritize completing both lecture drafts over polishing demos. Update lecture names on landing pages to match, but preserve the Unit 02 title **Working with Functions**. Do not rename the unit.
- Focus this round on `units/02/L06/index.qmd` and `units/02/L07/index.qmd`. Preserve the drafts' structure, voice, and mathematical visuals; the user is proud of them.
- Put concepts and physical interpretation first. Keep heavy mathematics and derivations in collapsed, expandable sections. Check for additional passages that need folding.
- For the current L06 refinements, the user has authorized direct edits to the local WASM notebook sources in `activities/`. Keep `student_map` in its own visible cell; collapse the other fixed-point demo code cells. If pairing on a live molab session later, use `marimo-pair` and edit through `marimo._code_mode`.
- Keep iteration short. Check the L06 notebooks and their numerical examples locally; leave full-site rendering to CI. Avoid long local Quarto renders or WASM/browser debugging.
- Use **Newton–Raphson** consistently in teaching prose and demo labels. Preserve required API names such as SciPy's `method="newton"`.
- Use `quarto preview` or `quarto render --to html` for daily HTML work. The shared hook exports local notebooks; `_quarto-full.yml` adds answer-PDF generation/staging and PDF format links. Complete local checks and CI use `quarto render --profile full` to build all HTML and PDF pages. The post-render hook generates `llms.txt`. Preserve full-site render targets and navigation in both paths.
- Original drafts, associated L06 figures, and the pre-focus `_quarto.yml` are saved in `_archive/lectures-06-07-20260915-160126/`, with SHA-256 checksums. The current full `_quarto.yml` was additionally checkpointed at `_archive/quarto-checkpoints/20260916-102601-before-l06-preview/`. Preserve these baselines.
- The archived `_quarto.yml` records the baseline rendering configuration. Preserve the current navigation, including S03, when changing build hooks.
- Molab session URL: `https://sb-93d3852d0dfbeff0.sb.molab.run/`. Use the token supplied in the conversation; do not save it in project files. Request a fresh connection if the session expires.

## Writing style

When the user asks for writing refinement, use two passes. The first pass may be drafted freely, but the second pass must proofread every sentence and remove garbage wording.

Avoid filler and defensive contrast patterns such as “this is X, not Y” or “this demo is only for X, not a real Y.” State the useful point directly. Preserve the user’s intended meaning, tone, and edits rather than replacing them with generic polished prose.

### Mandatory post-generation cleanup

Treat these as general failure patterns to find throughout a draft, not a blacklist of individual sentences. After drafting, read every sentence in its paragraph and every paragraph in the teaching sequence. Revise any passage that sounds like generic assistant advice rather than an instructor explaining the material to MSE students.

- **Disconnected short statements:** avoid chains of one- or two-sentence paragraphs that separately name a tool, add a caveat, and announce a demo. Develop the scientific idea in a connected paragraph, explaining how the example follows from it. Short questions, definitions, and equations can still stand on their own when that helps teaching.
- **Semicolons as artificial glue:** do not join short prose statements with a semicolon to imitate a scientific tone. Explain their actual relationship using natural connective language, or restructure the paragraph. This does not apply to mathematical notation or code.
- **Generic action slogans:** avoid headings and transitions built as “verb A, then verb B,” or empty directives about using tools, checking results, and making decisions. Name the method, physical question, numerical quantity, or specific comparison students are about to study.
- **Vague expandable-section titles:** a callout title must identify what opening it provides, such as a complete spline matrix, the derivation of a coefficient, or a worked divided-difference example. Do not use a title to postpone a topic or ask the reader to guess what is inside. Introduce why the example matters before the callout.
- **Compressed definitions:** introduce major paired concepts, such as interpolation and extrapolation, as clearly labelled bold definitions or bullet points with examples. Do not bury them in two nearly identical sentences.
- **Names without explanation:** naming a function family or algorithm is not an introduction. Give an accessible description or representative formula, explain what shape or physical situation it represents, and link to a useful reference when its details extend beyond the class. Keep students' curiosity open rather than dismissing a topic as unimportant.
- **Unexplained notation and computing jargon:** show sums/products in expanded form before relying on compact symbols. Explain subscripts, bracket notation, and which quantities are known. Describe the physical operation before introducing words such as query, lookup, or training data.
- **Defensive contrasts and disclaimers:** remove reflexive “this is A, not B” constructions and remarks that downgrade an example instead of explaining it. State the relevant assumption, purpose, or physical limit directly. Keep genuine mathematical distinctions and specific warnings, with the reason they matter.
- **Premature claims:** show the calculation or procedure before claiming why it is useful. For example, students should see how divided differences determine Newton coefficients before being told that new data can be appended efficiently.

A cleanup pass is complete only when the transitions explain why the next idea is needed, the essential mathematics remains visible, and the prose reads naturally as a lecture. Do not meet these rules by mechanically lengthening every paragraph or deleting all short sentences.

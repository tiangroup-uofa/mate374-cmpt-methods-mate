# Project instructions

## Templates

Use the matching scaffold in `templates/` when creating course pages:

| Page type | Template | Destination |
|---|---|---|
| Unit landing page | `unit-landing.qmd.template` | `units/NN/index.qmd` |
| Lecture note | `lecture-note.qmd.template` | `units/NN/Lxx/index.qmd` |
| Assignment and answer key | `assignment.qmd.template` | `assignments/AN/index.qmd` and `answer-keys/AN/answers.qmd` |
| Seminar | `seminar.qmd.template` | `seminars/Sxx-topic/index.qmd` |

Read the template, copy the relevant scaffold to its destination, replace placeholders, and remove authoring notes. Adapt the sections to the topic while preserving established course titles, tone, and navigation. For assignments, split the handout and answer-key sections into separate files. Templates remain outside HTML/PDF render targets.

## Adding course content

Before adding content, read its matching template and the relevant existing landing page; follow their structure and style. For every new page, update the appropriate landing-page links and site navigation in the root `_quarto.yml` (`website.sidebar` for unit and lecture entries, `website.navbar` for top-level categories), and confirm `project.render` includes it.

- **New lecture:** add it in sequence to its unit landing page.
- **New unit and its lectures:** create the unit landing page and lecture pages from their templates; add the unit and lecture links to `units/index.qmd` and a matching section in the sidebar. Follow the existing unit landing-page style.
- **New assignment:** add the corresponding student-facing marimo activity notebook source(s) under `activities/`, link or describe them in the assignment, and add the assignment to its landing page. Never ask students to use or run a plain Python script.

## Preview and render scope

- **Whole-site HTML:** `quarto preview` or `quarto render --to html`.
- **Complete local/CI build:** `quarto render --profile full`. This also regenerates figures and answer PDFs and enables PDF format links.
- **One-page test:** `quarto render units/NN/Lxx/index.qmd --to html`.
- **Temporary multi-page scope:** stop preview, save a copy of the full `_quarto.yml` outside the project, then temporarily replace only `project.render` in the local `_quarto.yml` with the desired page paths or globs. Keep navigation, resources, format definitions, and hooks intact. Run `quarto preview` or `quarto render --to html`.
- **Return to full scope:** restore the saved `project.render` list, restart preview, and use the whole-site or full-profile command above. Do not commit temporary scope changes or overwrite unrelated configuration edits.

`_quarto-full.yml` adds build hooks, not render targets: selecting `--profile full` alone does not undo a narrowed `_quarto.yml`. Profile render lists append to the base list, so a short `_quarto-local.yml` render list does not narrow the scope. Shared notebook-export hooks may still process notebooks outside the selected pages.

## Figure generation

All script-generated figures must use `dpi=300` when saved. Do not introduce lower-resolution output unless the instructor explicitly requests it.

## Assignment answer-key architecture

- Keep one answer-key source file per assignment: `answer-keys/AN/answers.qmd`. Store only the static PNG figures referenced by that file beside it. Keep answer prose, code, tables, and reported values directly in `answers.qmd`; avoid generated QMD fragments, results JSON, duplicate PDFs, and one-off extraction/check scripts in the answer-key folders.
- Use the existing rendered figures as the answer-key assets. Save all script-generated figures at 300 dpi; do not retain temporary figure-generation code solely to recreate an already-checked image unless the instructor asks for reproducibility.
- Keep editable completed marimo notebook sources in `activities/*_completed.edit.py`, never duplicated under `answer-keys/`. `scripts/export_completed_notebooks.py` builds the linked HTML notebooks into the ignored `completed-notebooks/` output directory. Treat that HTML and its assets as generated build output.
- Render and stage answer PDFs through `scripts/render_answer_pdfs.py` and the full Quarto profile. Do not commit rendered PDF copies inside `answer-keys/AN/`.

## Work in progress and planning

Keep work-in-progress notes, plans, and questions for the instructor in separate `.txt` files, never `.md` files, so they stay outside rendered course materials.

- **NEVER** insert planning questions, draft-status notices, or requests for instructor decisions as callouts or other text in course materials.
- **NEVER** use course text as a “schooling ground” to lecture, scold, or patronize the reader or instructor. Keep authoring commentary and editorial advice in the separate `.txt` notes. Student-facing text must teach the course subject respectfully.

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

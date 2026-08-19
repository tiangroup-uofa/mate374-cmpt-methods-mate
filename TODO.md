# MATE 374 reintegration: working plan and research backlog

> **Status:** initial strategy, not a frozen syllabus. This file is the durable inbox for course-design findings so later sessions do not need to rediscover the context.

## North star

MATE 374 should teach students to turn a materials question into a trustworthy computational prediction:

> **materials question → model and scale → mathematical problem → numerical representation → computation → verification → physical interpretation / decision**

This should not feel like “generic numerical methods first, assorted simulation packages later,” nor like CHE 374 with materials nouns substituted into its examples. Numerical methods should appear when a materials problem creates the need for them. Atomistic, mesoscale, and continuum viewpoints should recur throughout the term rather than form an isolated final block.

Provisional emphasis:

- **70%** model geometry, physical reasoning, verification, and interpretation
- **20%** numerical concepts and algorithm selection
- **10%** coding mechanics

A successful student should be able to formulate a problem, choose a method, identify approximations, test convergence/conservation, and reject an attractive but untrustworthy result. They need not reimplement every textbook algorithm.

## Recommendation in one page

### Recommended content architecture

Use **problem-centred units**, with methods introduced organically and simulation scales woven through them:

1. **Can I trust this number?** Approximation, floating point, error, convergence, reproducibility.
2. **What state does the material prefer?** Nonlinear equations, optimization, linear algebra; phase equilibrium and atomistic relaxation.
3. **What can sparse materials data tell us?** Regression, interpolation, differentiation, integration, uncertainty.
4. **How does a material change with time and space?** ODEs, conservation, finite differences/volumes, stability; kinetics, heat, and diffusion.
5. **How do microscopic rules produce macroscopic behavior?** Sampling, Monte Carlo, MD, periodicity, statistics, multiscale limits.
6. **How does structure determine materials properties?** Electronic-structure/QM foundations, structure–energy–property relationships, DFT as a computational example, and a modest introduction to data-driven property prediction.

### Recommended publishing architecture

Use a **hybrid single-repository system**:

- **Quarto is the course shell and printable source:** syllabus, unit narratives, concise lecture notes, equations, references, assignments, and PDF/HTML output.
- **marimo is the executable laboratory:** interactive demonstrations, guided investigations, student computational work, and self-contained WASM apps where feasible.
- **Light slides are views, not a second textbook:** reveal.js pages should contain prompts, diagrams, pivotal equations, and links/QR codes to live marimo activities. Avoid copying full notes into slides.
- Each class gets one durable landing page linking its printable note, activity, and after-class material. Organize navigation by **unit first**, class number second.

**Prototype status (2026-08-18):** MATE 374 keeps each micro-demo as a normal `.py` notebook and currently renders the same test through two format-aware Quarto filters. One compresses source into an editable `marimo.app/?embed=true&mode=edit#code/...` iframe. The other maps a filename relative to `activities/` to editable WASM HTML generated locally by the MATE 664-style post-render exporter; all local notebooks share one generated marimo asset directory. Both filters use the fenced Div contents as the static/PDF fallback. Compare appearance, load behavior, editing, responsive layout, and network requests before selecting a convention.

The same prototype page now supplies reading and classroom views from one rendered HTML source. Lecture pages marked `body-classes: lecture-page` receive an in-place URL switch between the ordinary reading URL and `?view=class`; class view hides navigation, expands the content from 938 px to 1372 px at a 1440 px viewport, and increases the base content font from 17 px to about 21 px. This replaces the old pattern of duplicating full notes, reveal.js slides, and handwritten-note links unless a genuinely different medium is needed.

Do **not** make marimo the only canonical format yet. It is excellent for live and student-facing computation, but long-form printing, page breaks, citations, cross-references, accessibility, and exam-study packets remain safer in Quarto/PDF. Also avoid maintaining independent full notes, full slides, and full notebooks: that creates three sources of truth.

### Authoring rule

For each class, author only what that medium does best:

- **Notes (Quarto):** stable explanation and derivation students will reread/print.
- **Slides (light):** classroom pacing, questions, visuals, and transitions.
- **Notebook (marimo, only when computation adds value):** manipulable model, experiment, visualization, or exercise.

A class does not automatically require all three. “No notebook needed” and “no slides needed” are valid decisions.

## Delivery-model options and weights

Weights reflect fit to pedagogical goals, maintainability, print quality, classroom interactivity, and migration cost.

| Option | Description | Weight | Verdict |
|---|---|---:|---|
| **A. Quarto spine + marimo computational labs + light slides** | Quarto owns narrative/print; marimo owns executable experiences; slides are deliberately sparse | **60%** | Recommended. Best balance and builds directly on CHE 318/MATE 664 experience. |
| **B. Marimo-first, export everything else** | Every lesson is a marimo notebook; HTML/PDF/static views generated from it | **25%** | Worth prototyping for one unit. Attractive single-source ideal, but print/citation/layout and notebook-length concerns need evidence before committing. |
| **C. Quarto-first with embedded or linked micro-demos** | Keep the established two-course pattern; marimo is supplemental | **12%** | Lowest migration risk, but likely reproduces the current authoring burden and underserves the “playground” goal. |
| **D. Separate full slide deck + full notes + full notebook per class** | Three parallel products | **3%** | Reject except for a few showcase classes; maintenance cost and drift are too high. |

### Decision gate before full production

Build one complete vertical slice (suggested: diffusion/conservation) in both A and B. Compare:

- print/PDF legibility and page breaks;
- time to revise one equation or figure across outputs;
- classroom presentation quality;
- accessibility and mobile behavior;
- student launch friction, offline/WASM behavior, package limits;
- ability to provide a non-executable study copy;
- instructor prep time.

Unless marimo-first clearly wins this test, adopt Option A for the first offering.

## Proposed units and rough class map

Assumption: approximately 35 × 50-minute meetings, matching Fall 2025. This map intentionally alternates concepts and applications. It can compress to 32–34 meetings.

### Unit 0 — Orientation: computation as a materials experiment (Classes 1–2)

**Driving question:** What does it mean for a simulation to be useful and believable?

1. A computational materials workflow; scales and representations; one result viewed through DFT/MD/MC/phase-field/FEM/data models.
2. Reproducible Python/marimo workflow; units, arrays, plotting, records of assumptions; diagnostic rather than a Python boot camp.

Anchor activity: inspect a polished-but-wrong simulation and list the evidence needed to trust it.

### Unit 1 — Can I trust this number? (Classes 3–5)

3. Approximation taxonomy: model-form, parameter/data, discretization/truncation, finite precision, iterative/solver tolerance, sampling error.
4. Floating point and conditioning through an equation-of-state derivative; U-shaped error versus step size; add “SCF noise.”
5. Verification ladder: exact/limiting cases, manufactured or benchmark solutions, mesh/timestep refinement, conservation, independent implementations; error-story studio.

Possible stories from the shared conversation: Sleipner A (discretization/FEM), Vancouver Stock Exchange (biased quantization), Patriot timing (finite representation), Fast16/other failure examples. Carefully source and distinguish error types before teaching.

### Unit 2 — What state does the material prefer? (Classes 6–11)

6. Root finding from equilibrium vacancy concentration or electroneutrality: bracket first, then solve.
7. Nonlinear systems through phase coexistence/equality of chemical potentials; initial guesses and multiple roots.
8. Free-energy minimization and common tangents: roots versus optimization; local/global minima and constraints.
9. Atomistic relaxation: energy landscape, forces as gradients, convergence criteria; compare a toy pair potential with a library optimizer.
10. Linear systems as coupled balances/spring or diffusion networks; matrix geometry, rank, residual, and conditioning.
11. Sparse/banded systems from local interactions; direct versus iterative solvers; “glass box once, library afterward.”

Suggested anchor: a binary regular-solution free-energy notebook connects root finding, differentiation, optimization, phase diagrams, and sensitivity in one evolving model.

### Unit 3 — What can materials data tell us? (Classes 12–16)

12. Regression with Arrhenius diffusivity data: transformations, residuals, parameter meaning, uncertainty.
13. Interpolation versus modeling: property tables/CALPHAD-like data, splines, extrapolation hazards.
14. Numerical differentiation: pressure from E(V), stress/elastic quantities, noise amplification; sampled data versus callable models versus automatic differentiation.
15. Numerical integration: work/toughness from stress–strain data, enthalpy/heat capacity, radial distribution functions.
16. Model calibration and validation studio: identifiability, residuals, uncertainty, and overfitting. If calendar coverage requires ML, use only a brief library-level property-fitting example here; do not add neural networks, GNNs, or LLMs to the core course.

### Unit 4 — How does a material evolve? Lumped models (Classes 17–21)

17. From physical inventory to ODE: precipitation fraction, grain growth, oxidation, or lumped thermal history.
18. Euler as a numerical experiment; local/global error and timestep refinement.
19. Stability and stiffness through fast/slow materials kinetics; explicit versus implicit thinking.
20. Established integrators (`solve_ivp`) and event handling; solver tolerances are not physical accuracy.
21. Coupled ODEs and inverse kinetics; compare competing mechanisms or fit kinetic parameters.

MD foreshadowing: use oscillator integration here to ask why a mathematically converged trajectory can have unacceptable energy drift.

### Unit 5 — How does change move through a material? Fields and conservation (Classes 22–28)

22. Field → gradient → flux → divergence/conservation; diffusion and heat as the same numerical skeleton but different materials questions.
23. Geometry of a grid: cells, nodes, faces, boundary normals, Dirichlet/Neumann/mixed boundaries.
24. Derive and implement one finite-difference/finite-volume stencil deeply; sparse matrix emerges naturally.
25. Transient diffusion and method of lines; explicit timestep stability and diffusion length.
26. Variable diffusivity and interfaces: why `D*laplacian(c)` differs from `divergence(D*gradient(c))`; conservation tests.
27. Implicit solve and sparse linear algebra revisited; accuracy versus stability.
28. Phase-field bridge: Cahn–Hilliard or Allen–Cahn as free energy + kinetics + operators; what the numerical grid can and cannot resolve.

Adopt the shared-conversation principle: **one stencil deeply, then stop making students rewrite stencils**. Consider a tiny transparent `mate374.operators` layer (`gradient`, `divergence`, `laplacian`, `integrate`) that students inspect/test once and then use. Do not build it until the learning API is specified.

### Unit 6 — How do microscopic rules become properties? (Classes 29–33)

29. Probability, ensembles, observables, sampling error; Monte Carlo integration as an experiment.
30. Metropolis Monte Carlo with lattice ordering/phase transformation; equilibration, autocorrelation, finite-size effects.
31. Molecular dynamics as ODE integration: force, velocity Verlet, timestep, conservation, energy drift.
32. Periodic boundaries, minimum image, cutoffs, neighbor interactions; radial distribution and diffusion from MSD.
33. Thermostats/ensembles and LAMMPS: software as an instrument; reproduce one transparent toy result before using the package.

MC and MD should not be “software tourism.” Each should reinforce an earlier numerical idea: convergence/sampling for MC, time integration/stability for MD, optimization for relaxation, integration/statistics for observables.

### Unit 7 — How does structure determine materials properties? (Classes 34–35)

34. **Where does the interatomic energy come from?** Adapt Tian's 2025 guest lecture: Lennard–Jones and EAM → experimental calibration and virial validation → many-electron Schrödinger equation → Born–Oppenheimer approximation → density and Kohn–Sham DFT → self-consistency, XC approximation, basis/cutoff, k-points, and structure inputs.
35. **How do we make accurate energies affordable?** Continue the same narrative: fit a transparent LJ potential to DFT data (Kitchin/ASE example) → larger first-principles datasets → ML interatomic potentials as a modern endpoint. Explain atomic representations, locality, energies/forces, computational cost, and trust at a high level; do not teach neural-network or GNN implementation.

Verification, validation, uncertainty, and reproducibility remain recurring practices in every unit rather than becoming a detached final synthesis unit. If a project is used, its checkpoints should run throughout the term instead of occupying Unit 7.

## Assessment direction

The 2025 scheme (25% assignments, 30% midterm, 45% final) rewards a conventional methods course more than an end-to-end computational workflow. Consider:

- 20–25% short concept/problem sets;
- 20–25% reproducible computational investigations;
- 20–25% midterm;
- 25–35% staged final project or practical synthesis;
- optional small participation/verification audits.

Every substantial computational submission should include:

1. question and model;
2. assumptions and scales;
3. numerical method and why it fits;
4. at least one verification/convergence test;
5. physical interpretation with units;
6. limitations and reproducibility information.

Prefer rubrics where solver choice, evidence of convergence, and physical interpretation outweigh reproducing an algorithm from memory.

Possible semester project: one end-to-end study—fit or select a model, solve a transport/kinetic/equilibrium/atomistic problem, demonstrate numerical credibility, compare against data/analytic limits/literature, and communicate a materials conclusion. Require proposal → checkpoint → final artifact to prevent a last-week package demo.

## What to reuse

### Fall 2025 MATE 374

Source: `~/Google Drive/My Drive/Teaching/MATE 374/Fall 2025 (Hao Zhang)/`

Keep/reference:

- official calendar scope and learning outcomes;
- 34-lecture coverage map;
- Tian's 2025 marimo guest lecture, `guest_lecture/mate374_guest_lecture_Tian.py`, as the primary source for Unit 7. Its existing conceptual spine—LJ → EAM → Schrödinger/Born–Oppenheimer → KS-DFT/self-consistency/XC → fit LJ to DFT → MLIPs—is much better than introducing a new generic structure–property or ML lesson. Split and simplify it across the final two classes, preserve the interactive virial/LJ idea, and correct/test technical details before reuse;
- assignments and hands-on sheets as a question bank;
- LAMMPS workstation instructions;
- lectures 1–3 for error/floating-point raw material;
- lectures 4–25 for conventional method examples, selectively compressed;
- lectures 26–34 for simulation-scale and MD/MC raw material;
- Tian guest lecture marimo source as an existing MATE 374 prototype.

Major issue: the term is visibly split into a long CHE 374-derived methods sequence followed by “materials simulation,” with many classes devoted to variants of interpolation, quadrature, elimination, and ODE algorithms. Reintegrate rather than merely shorten the first half.

### CHE 318

Source: `~/Dev/che318-mass-transfer/`

Reuse the proven production pattern:

- Quarto website, per-lecture folders, reveal.js lecture pages;
- marimo script gallery/export pipeline;
- deployment workflow;
- concise per-class learning outcomes and recurring recap/summary rhythm;
- existing diffusion/conservation and boundary-condition demos.

Avoid inheriting the assumption that every class needs a long slide-like `index.qmd`. Audit authoring time and duplication.

### MATE 664

Source: `~/Dev/mate664-kinetics-of-materials/`

This is the richest local content donor:

- L04–L10: diffusivity, diffusion equations, finite difference, MSD/random walk, activated atomic models;
- L13–L17: free-energy diagrams, nucleation, spinodal decomposition, Cahn–Hilliard, coarsening;
- L20: continuum/phase-field simulation workflow;
- L21: Monte Carlo, Potts, and KMC;
- L22: MD, Verlet, timestep, ensembles, potentials;
- scripts: Arrhenius, diffusion length, finite-difference comparison, interdiffusion fitting, Cahn–Hilliard, heterogeneous nucleation;
- assignments as possible advanced examples to simplify for MATE 374.

Reuse models/figures/code, but rewrite for third-year numerical learning outcomes rather than transplanting graduate-level derivations.

### Other local material discovered

- `~/Dev/mate374-test/gauss_elimination_geometry.py` — live marimo experiment on geometric interpretation of Gaussian elimination. Evaluate as Unit 2 prototype.
- Current repo began empty, so architecture is unconstrained.

## External inspirations to audit

Do not clone these courses wholesale; make an “idea card” for each useful example: materials question, numerical concept, prerequisite, adaptation cost, interactive opportunity, and license/source.

- John Kitchin’s CMU computational engineering / Python materials and chemical engineering notebooks and books.
- Zachary Ulissi’s CMU numerical methods/Jupyter course materials.
- SciPy lecture notes and scientific Python tutorials for modern library usage.
- Materials Project/ASE tutorials for structure, equation-of-state, and atomistic workflows.
- pycalphad examples for equilibrium/common-tangent ideas (possibly too heavy for browser use).
- FiPy/py-pde examples for phase-field and diffusion (inspiration; avoid hiding discretization too early).
- LAMMPS examples and educational resources for a reproducible MD case.
- Software Carpentry only for minimal environment/reproducibility patterns, not as course content.

## Shared ChatGPT conversations reviewed

1. `6a7c09a9-f8c4-83e8-90e3-f4c1e4730716` — error-analysis ideas: EOS numerical derivative and U-shaped error curve; distinguish truncation/discretization, roundoff, solver noise/tolerance, and model error; real failure stories.
2. `6a7c0a89-bed8-83e8-aaaf-17fa3da82c55` — prior project/course revamp estimate and staged “vertical slice first” migration strategy.
3. `6a7c0ab4-4b84-83e8-9da4-f03621f53220` — materials-centred method map, 70/20/10 emphasis, “glass box once” policy, operator-layer idea, verification-centred outcomes.

The first conversation title shown by ChatGPT is unrelated (“Fast16…” in Chinese), but its content includes relevant numerical-error teaching ideas; retain the URL and verify exact context before citing it publicly.

## Course-design guardrails

- Start from a **question**, not an algorithm name.
- Revisit the same model across units when possible (regular solution, diffusion couple, Lennard-Jones solid) so students see connections rather than 35 unrelated examples.
- Teach each core mechanism transparently once; afterward use trusted libraries and focus on selection/diagnosis.
- Separate **verification** (“did we solve the equations right?”) from **validation** (“are these equations adequate for reality?”).
- Always distinguish model, parameter/data, discretization, iterative, floating-point, and sampling uncertainty.
- Require units, limiting behavior, conservation, and convergence plots.
- Do not turn the course into Python syntax training or a tour of branded packages.
- **ML scope cap:** MATE 374 is a computational/numerical methods course, not a materials-informatics course. At most, reserve one class or one optional demonstration for elementary supervised property fitting if required by the official calendar. Feature engineering, neural networks, GNNs, ML potentials, generative models, and LLM workflows belong elsewhere unless the approved syllabus is formally changed.
- Do not imply that one lecture provides competence in DFT, FEM, ML, phase field, MD, and MC. Teach a transferable simulation workflow and honest scale/method selection.
- Prefer NumPy/SciPy/browser-compatible examples for access, but do not let WASM compatibility dictate all scientific choices. Heavy-code experiences can be instructor-run or workstation-based.
- Preserve printable, static fallbacks for every assessed activity.

## Repository layout proposal

```text
MATE374/
├── README.md
├── TODO.md                         # this durable planning inbox
├── _quarto.yml
├── pyproject.toml
├── index.qmd
├── syllabus/
├── units/
│   ├── 00-orientation/
│   ├── 01-trust/
│   ├── 02-equilibrium/
│   ├── 03-data/
│   ├── 04-odes/
│   ├── 05-fields/
│   ├── 06-micro-to-macro/
│   └── 07-synthesis/
│       └── Lxx/
│           ├── index.qmd           # printable note / landing page
│           ├── slides.qmd          # optional, light
│           └── activity.py         # optional marimo notebook
├── activities/                     # cross-unit or standalone labs
├── assignments/
├── project/
├── mate374/                        # optional tiny transparent helpers
├── data/
├── assets/
└── references/
```

Keep source notebooks as normal marimo `.py` files. Exported HTML/PDF belongs in build output, not alongside canonical source unless deployment requires it.

## Immediate work plan

### Phase 1 — Evidence and decisions

- [ ] Confirm calendar: class count, seminar/lab structure, prerequisites, required official learning outcomes, software/workstation constraints.
- [ ] Extract a complete inventory of all 2025 lecture/assignment topics and tag each **retain / compress / replace / relocate**.
- [ ] Compare current MATE 374 with current CHE 374 formally, not only inherited slide labels.
- [ ] Review all 2025 assignments/solutions and hands-on sheets; build a method × materials-context × cognitive-level matrix.
- [ ] Audit `~/Dev/mate374-test/gauss_elimination_geometry.py`.
- [ ] Audit the 2025 guest lecture marimo notebook and note what worked in class.
- [ ] Search local Google Drive/Dev/Documents more broadly for MATE 374 notes, project ideas, DFT/FEM/LAMMPS teaching assets, and old conversations.
- [ ] Review Kitchin and Ulissi materials and record only high-value idea cards with links/licenses.
- [ ] Decide assessment constraints with the department before designing all activities.

### Phase 2 — Prototype the system

- [ ] Select one vertical slice: recommended Unit 5 diffusion/conservation.
- [ ] Build Option A version: Quarto note + light slides + marimo investigation + printable activity.
- [ ] Build a marimo-first Option B version of the same slice.
- [ ] Time revisions and test print, browser, mobile, WASM/offline, accessibility, and clean-machine setup.
- [ ] Ask 2–3 students/TAs to test launch and print workflows.
- [ ] Freeze authoring conventions only after this comparison.

### Phase 3 — Freeze curriculum architecture

- [ ] Write 6–8 assessable course learning outcomes centered on formulation, selection, verification, interpretation, and reproducibility.
- [ ] Finalize units and class allocation.
- [ ] Choose 2–3 recurring anchor models across the term.
- [ ] Design the project, milestones, and rubric before polishing lectures.
- [ ] Define “glass box once” boundaries: which algorithms students implement, inspect, or only call.
- [ ] Define minimum Python diagnostic/remediation route.

### Phase 4 — Build first offering

- [ ] Establish repository, Quarto navigation, environment, CI/deployment, notebook export, and PDF pipeline.
- [ ] Produce Units 0–2 fully before bulk migration.
- [ ] Port reusable MATE 664 marimo models with simplified narrative and explicit verification tasks.
- [ ] Create structured placeholders/source links for later units rather than prematurely polishing every class.
- [ ] Add automated smoke tests for notebooks, links, Quarto render, and key numerical invariants.

## Open decisions

- [ ] Which year/term will Tian first teach the redesigned course?
- [ ] Is the official 3-1S-0 seminar available for computational studios, or must all hands-on work fit lectures?
- [ ] Must DFT, FEM, ML, phase field, MC, and MD each be explicitly assessed due to calendar/accreditation expectations?
- [ ] Is a final project administratively possible, or must the final exam retain a large fixed weight?
- [ ] What do students already learn in ENCMP 100 and CHE 374, and do MATE students take CHE 374 at all?
- [ ] Student device policy: browser-only target, local Python, managed lab, or mixed?
- [ ] Is printing needed per class, per unit packet, or only for a course notes book?
- [ ] Should slides be public, and should instructor-only notes/solutions live in a private companion repo?
- [ ] How much graduate MATE 664 content can be reused without undermining that course’s distinct role?
- [ ] Are there licensing/permission constraints on Hao Zhang/Nikrityuk inherited notes and external notebooks?

## Parking lot / future ideas

- A course-wide “simulation credibility checklist” printed on every assignment.
- Error-budget notebook using E(V) → pressure, with exact, floating-point, and synthetic SCF-noise layers.
- One regular-solution notebook that grows from fitting → roots/common tangent → spinodal → Cahn–Hilliard → Monte Carlo.
- One Lennard-Jones notebook that grows from energy/force → minimization → Verlet → ensembles → RDF/MSD.
- “Same diffusion problem, four representations”: random walk, PDE, finite-volume computation, atomistic MSD.
- Deliberately broken notebook practical: wrong units, unstable timestep, incorrect boundary condition, non-converged optimization, misleading fit.
- Tiny transparent operator layer, but only if it reduces syntax without concealing variable placement and boundary fluxes.
- Generate a printable end-of-unit “concept map + equations + diagnostics” rather than printing interactive notebooks verbatim.
- Use marimo sliders sparingly: every interaction should expose sensitivity, a regime transition, stability, or uncertainty—not merely animate a curve.

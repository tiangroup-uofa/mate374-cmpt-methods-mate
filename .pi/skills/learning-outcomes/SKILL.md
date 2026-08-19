---
name: learning-outcomes
description: Write, audit, and align measurable MATE 374 learning outcomes using Bloom’s taxonomy. Use when defining course, unit, class, notebook, activity, or assessment outcomes and when checking that teaching and evaluation align.
---

# MATE 374 learning outcomes

Read `TODO.md` and the relevant unit plan first. Outcomes must reflect numerical and computational materials engineering, not generic Python fluency or software operation.

## Workflow

1. Identify the level: whole course, unit, class, activity, or assessment.
2. Identify the materials context and the observable student performance.
3. Select an appropriate Bloom level: Remember, Understand, Apply, Analyze, Evaluate, or Create. Do not force every outcome into Create.
4. Write the outcome with an observable verb, object, conditions, and—when useful—a quality criterion.
5. Check progression: class outcomes support unit outcomes, and unit outcomes support the course outcomes.
6. Map each outcome to evidence: prediction, calculation, code/notebook change, diagram, explanation, diagnostic, convergence study, or design decision.
7. Remove outcomes that are only activities (“use a slider”) or vague mental states (“understand DFT”).

## Preferred outcome form

> Given **[materials problem/data/model]**, students will **[observable verb + object]** using **[method/condition]**, and justify the result with **[numerical or physical evidence]**.

Use one main verb per outcome where possible. Keep a class to roughly 2–4 outcomes and a unit to roughly 4–7. Course-level outcomes should remain few enough to assess honestly.

## Bloom guidance

- **Remember:** identify error sources, define residual, recall stability condition.
- **Understand:** explain why a discretization or ensemble represents a question.
- **Apply:** compute an equilibrium, fit a model, integrate an ODE, or assemble a stencil.
- **Analyze:** diagnose an unstable timestep, separate error sources, interpret residuals, or compare model assumptions.
- **Evaluate:** defend a solver or simulation result using convergence, units, conservation, uncertainty, and physical limits.
- **Create:** formulate and document a small reproducible materials simulation study.

Higher Bloom levels should not mean harder algebra for its own sake. They should represent engineering judgment.

## Alignment table

When auditing or producing outcomes, include:

| Outcome | Bloom level | Learning evidence | Teaching activity | Assessment form |
|---|---|---|---|---|

Make sure each important outcome has an activity and an assessment opportunity. Flag outcomes that are taught but never evidenced, or assessed without preparation.

## MATE 374 emphasis

Favor outcomes involving:

- formulation from a materials question;
- selection of a numerical representation or solver;
- assumptions, scales, units, and boundary/initial conditions;
- implementation or inspection of a core method;
- verification, validation, uncertainty, and reproducibility;
- interpretation of results as a materials engineering decision.

ML, DFT, MD, Monte Carlo, phase field, and other tools should be framed honestly at the level actually taught. Do not claim competence in a production method from a conceptual demonstration.

When asked to revise outcomes, preserve the approved course scope and identify any outcome that would require a syllabus change.

---
name: teaching-outline
description: Plan or revise MATE 374 units and class meetings as concise, active-learning engineering lessons. Use when allocating content across classes, designing an in-class sequence, integrating modern computational tools, or reviewing whether a lesson is teachable.
---

# MATE 374 teaching outline

Treat MATE 374 as a numerical and computational methods course for materials engineers. Read `TODO.md` and relevant unit/source materials before planning. Preserve the current seven-unit structure and the course loop:

> materials question → model → numerical representation → computation → verification → physical interpretation

## Workflow

1. Identify the unit, class number, available meeting time, prerequisites, official constraints, and the materials question.
2. Decide what students must *do* by the end of the class. Limit the class to one central idea and a small number of supporting techniques.
3. Allocate content across classes. Do not force every inherited topic, algorithm, software package, or notebook into the schedule.
4. Build an active-learning sequence before writing exposition.
5. Choose the lightest tool that improves the learning: Quarto for durable notes, an embedded marimo iframe for computation, and tldraw for live spatial reasoning or annotation. Do not add a tool merely because it is modern.
6. Include a credibility check: units, limiting case, residual, convergence, conservation, uncertainty, or comparison with an analytical/reference result.
7. State what evidence will show that students learned the idea and what should be revisited in the next class.
8. Mark assumptions and unresolved calendar decisions rather than inventing official requirements.

## Default 50-minute rhythm

Adapt this rather than following it mechanically:

- 0–5: retrieval, prediction, or provocative materials observation;
- 5–12: materials context and learning target;
- 12–20: short explanation or derivation;
- 20–35: marimo investigation, worked example, or paired reasoning task;
- 35–45: compare results and diagnose numerical/physical differences;
- 45–50: credibility check, exit question, and bridge to the next class.

Use several short cycles for difficult topics instead of one long lecture. Put the most important reasoning before the tool interaction, not after it.

## Required outline fields

Return a compact outline containing:

- unit and class title;
- driving materials question;
- 2–4 measurable learning outcomes, linked to Bloom levels;
- prerequisite knowledge and likely misconceptions;
- essential model/equations and numerical method;
- timed active-learning sequence;
- materials-engineering example and interpretation prompt;
- tool plan: Quarto, marimo iframe, tldraw, or no special tool;
- verification/validation or uncertainty check;
- evidence of learning and a short exit question;
- printable/static fallback and accessibility considerations;
- what to omit or defer;
- instructor reflection questions for after class.

## Guardrails

- Start with a materials question, not “today we learn Newton’s method.”
- Distinguish model error, data/parameter error, discretization error, iterative error, floating-point error, and sampling error.
- Teach an important algorithm transparently once; subsequently emphasize method selection, diagnostics, and interpretation.
- Keep Python syntax subordinate to engineering reasoning.
- Do not make students complete a separate tldraw or marimo artifact unless it has a clear learning purpose.
- Prefer a small, discussable activity over a polished but passive demonstration.
- Preserve a printable Quarto version even when the live lesson uses embedded interactive content.

If information is missing, make a clearly labeled assumption and give one alternative allocation rather than stopping for unnecessary clarification.

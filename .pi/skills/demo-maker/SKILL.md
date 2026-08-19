---
name: demo-maker
description: Design, create, refactor, and validate clear MATE 374 marimo demonstrations intended for embedded browser use, preferably WASM-compatible. Use when building a notebook for a Quarto iframe, converting a teaching idea into an interactive demo, or checking notebook clarity, reactivity, and export readiness.
---

# MATE 374 demo maker

Create marimo notebooks as focused teaching instruments, not miniature software packages. Read the relevant unit/class outline and `TODO.md` first. The notebook should answer one materials question and expose one important numerical idea.

## Workflow

1. Define the materials question, audience, prerequisite, expected class duration, and one central learning claim.
2. Choose the smallest model that makes the claim visible. State assumptions, units, scales, and known limitations.
3. Decide what students should predict, manipulate, observe, and explain before writing code.
4. Design the notebook in this order:
   - question and prediction;
   - model and parameters;
   - transparent reference implementation;
   - interactive controls;
   - visualization or numerical result;
   - credibility check;
   - interpretation and extension.
5. Keep cells small, acyclic, and readable. Let marimo reactivity carry dependencies; avoid hidden mutable state and unnecessary guards.
6. Keep widgets visible in both interactive and script modes. Use deterministic defaults and synthetic/small data so clean-machine checks can run.
7. Build a static fallback: representative figure, essential equation, key result, and a short explanation that can appear in Quarto/PDF when the iframe is unavailable.
8. Validate with `marimo check` and a non-interactive execution. Test the intended embed/export path, browser behavior, load time, and failure mode before calling it classroom-ready.

## Notebook contract

Every demo should make these blocks visible:

1. **Materials question** — what is being predicted or explained?
2. **Prediction** — what should change and why?
3. **Model** — equation, energy, probability law, or data relation.
4. **Numerical method** — what is discretized, sampled, optimized, or integrated?
5. **Experiment** — controls expose sensitivity, a regime transition, stability, uncertainty, or convergence.
6. **Trust check** — units, limiting case, residual, conservation, convergence, or reference comparison.
7. **Interpretation** — what does the result mean for the material, and what does it not mean?

## WASM and embedding priorities

Prefer NumPy, SciPy functionality known to work in the target browser environment, matplotlib/plotly paths already supported by the project, and small in-memory datasets. Avoid filesystem access, subprocesses, network calls, native-only scientific packages, large downloads, secrets, and long-running calculations in the default path. If native execution is scientifically necessary, provide a browser-safe conceptual path or a precomputed-data fallback and label the limitation.

Use the Quarto page as the integration point. The marimo app is embedded with an iframe; do not require students to follow a separate notebook link during class. Include an iframe title, sensible dimensions, loading behavior, and a print/static fallback. Keep the notebook source in the repository and generated artifacts separate according to the project convention.

## Interaction rules

- Every control must answer “what changes when I move this, and why?”
- Use defaults that show an interpretable baseline.
- Limit the number of simultaneous controls.
- Prefer one parameter change at a time for causal reasoning.
- Make units and parameter ranges visible.
- Avoid animation and decorative sliders.
- Do not hide the numerical method behind a black box when that method is the lesson.
- Use trusted library calls after students have inspected the core mechanism once.

## Validation checklist

Before reporting completion, check:

- the file has valid marimo structure and declared dependencies;
- `marimo check <notebook.py>` passes, or the exact blocker is recorded;
- script/non-interactive execution completes with deterministic defaults;
- no accidental cell cycles, hidden state, or mutation-driven behavior;
- widgets and final cell outputs render;
- the result has units and a visible credibility check;
- the notebook is understandable without live narration;
- the Quarto iframe and static fallback are both specified;
- the WASM/native compatibility decision is explicit;
- likely classroom failure modes and recovery steps are documented.

Do not add tests, helper packages, or elaborate UI unless they support the learning objective. A short, reliable demo is better than a feature-rich fragile one.

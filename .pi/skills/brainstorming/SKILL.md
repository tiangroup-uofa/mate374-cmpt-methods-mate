---
name: brainstorming
description: Generate, refine, and classify MATE 374 teaching questions for improvisation, active discussion, polling, or Wooclap. Use when mining course materials for misconceptions, prediction prompts, multiple-choice questions, ranking tasks, or short engineering questions.
---

# MATE 374 brainstorming

Use the course materials as the source, but turn inherited examples into question-first materials-engineering prompts. Read the relevant notes, notebook, assignment, or unit plan before generating questions.

## Workflow

1. Identify the target unit/class, concept, prerequisite level, available response time, and intended medium: live improvisation, Wooclap poll, worksheet, or exit ticket.
2. Extract the engineering decision or numerical idea students should reason about.
3. Identify likely misconceptions, seductive wrong answers, hidden assumptions, and the diagnostic evidence that distinguishes them.
4. Generate several question types before selecting the best one:
   - prediction before computation;
   - single- or multiple-choice concept question;
   - numerical estimate with units;
   - ranking or ordering;
   - matching method to materials question;
   - interpret-a-plot/table/diagram;
   - error diagnosis or “which result should we trust?”;
   - short justification or compare-and-defend prompt.
5. Select questions that create useful discussion, not trivia. Prefer questions where a wrong answer reveals a specific misconception.
6. Provide the answer, reasoning, distractor diagnosis, Bloom level, estimated response time, and the follow-up discussion move.
7. Check wording, units, uniqueness of the answer, accessibility, and whether the question can work without fragile software.

## Output format

For each selected question, return:

- **Question / prompt**
- **Format and timing**
- **Materials context**
- **Options or response instruction**
- **Expected answer**
- **Why it is correct**
- **What each important wrong answer diagnoses**
- **Bloom level**
- **Follow-up:** what the instructor should ask or show next
- **Optional marimo/tldraw connection**

Offer a compact Wooclap-ready version separately from instructor notes. Keep the student-facing version free of answer cues.

## Quality rules

- Ask about model choice, numerical behavior, evidence, or physical interpretation—not Python syntax.
- Use realistic materials quantities and units, but avoid arithmetic that overwhelms the concept.
- State enough information for a unique answer; if ambiguity is pedagogically intentional, say so.
- Avoid trick questions and “all of the above” unless the reasoning genuinely requires it.
- Include a prediction before a plot or simulation whenever possible.
- Do not turn every question into multiple choice; open justification is often more diagnostic.
- Connect follow-up discussion to verification, validation, conservation, limiting behavior, uncertainty, or scale.
- Preserve the distinction between a mathematically converged answer and a physically appropriate model.

For improvisation, give the instructor a one-line prompt, likely student responses, and a recovery question. For Wooclap, keep response choices short and mobile-friendly.

# Lecture-note format A/B test

## First review decision

The preferred direction is the L03 structure: a central-question callout followed by a limited number of broad sections. The boxed statements from L04 should be retained for important equations, contrasts, and synthesis chains. L04 has been revised in this hybrid style, and `templates/lecture-note.qmd` is now the working template.

The original A/B comparison is retained below as design history.

The first four rewritten lectures used two text-first structures while holding the visual system and content density roughly constant.

## Variant A: chapter-first

Used in L01 and L03. It opens with a central-question callout, then develops a continuous chapter with recurring conceptual sections. It ends with “What to keep” and several checks.

Strengths to test:
- reads naturally as a small textbook chapter;
- central argument is visible;
- works well for concept-heavy lectures.

Possible weakness:
- the in-class route is less explicit.

Template: `templates/lecture-note-a.qmd`.

## Variant B: route-first

Used in L02 and L04. It opens with “The question” and a short route through the problem. The lecture then progresses through a sequence of comparisons and ends with one synthesis test.

Strengths to test:
- easier to navigate while teaching;
- makes the progression and comparison explicit;
- works well for experiment-driven lectures.

Possible weakness:
- can feel more like structured teaching notes than a chapter.

Template: `templates/lecture-note-b.qmd`.

## Decisions to make after review

1. Which format is easier to read after class?
2. Which format is easier to scroll through while teaching?
3. Should “Learning outcomes” stay near the top or move to a compact opening box?
4. Should every page end with several short questions or one larger synthesis question?
5. Should optional history and implementation detail use collapsed callouts?

Before treating the hybrid as final, review revised L04 in ordinary and class views, then apply the same heading hierarchy to L01 and L02 without mechanically forcing identical sections.

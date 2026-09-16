# S03 · When does repeated updating settle?

> Earlier planning outline, retained for reference. The student draft is now in `index.qmd`; it uses the CO₂ vdW equation from the current L07 demo and includes a square-root example using Heron's method. The original outline below predates that choice.

**Subtitle:** Fixed-point iteration, cobweb diagrams, and convergence

## Recommendation

Make this a focused, paper-first seminar on fixed-point iteration. The mathematical idea is important, but the activity can remain light, visual, and practical. It connects L05, L06, and L07 without introducing another large method.

Reuse the existing L06 fixed-point notebook for the interactive part rather than building a new computational model.

## Central question

> If we repeatedly apply
> 
> $$
> x_{k+1}=g(x_k),
> $$
> 
> why does the sequence sometimes settle down and sometimes move away?

## Learning outcomes

By the end of the seminar, students should be able to:

1. Trace a fixed-point iteration by hand.
2. Distinguish a fixed point from a root of the original equation.
3. Predict convergence, alternating convergence, or divergence from a cobweb diagram.
4. Check an apparent solution using the original residual.

## Paper-first example

Begin with the simple root-finding problem

$$
 f(x)=x-2=0.
$$

All three maps below have the same fixed point, $x_s=2$:

| Iteration map | Starting at $x_0=0$ | Behaviour |
|---|---|---|
| $g_1(x)=1+\tfrac12x$ | $1,\ 1.5,\ 1.75,\ldots$ | Approaches from one side |
| $g_2(x)=3-\tfrac12x$ | $3,\ 1.5,\ 2.25,\ldots$ | Alternates toward the fixed point |
| $g_3(x)=1.2x-0.4$ | $-0.4,\ -0.88,\ -1.456,\ldots$ | Moves away |

Students calculate the first four or five iterates, then sketch $y=g(x)$ and $y=x$. The slopes $0.5$, $-0.5$, and $1.2$ give the visual rule:

- $|g'(x_s)|<1$: nearby iterates contract toward the fixed point;
- a negative slope produces alternating iterates;
- $|g'(x_s)|>1$: nearby iterates move away.

No contraction proof is needed.

## Return to the L06 materials problem

Use the existing L06 fixed-point notebook for the second half. Students compare:

- $g(x)=x-0.10f(x)$, which approaches a root;
- $g(x)=x+f(x)$, which moves away;
- the rearrangement $c(x)=-x/2$, which remains at $x=0$.

The key credibility check is

$$
\text{small step}
\quad\text{does not necessarily mean}\quad
\text{small residual}.
$$

For the last map, the iteration stays at $x=0$, but $f(0)=-1$. Students should identify this as a fixed point of the iteration map that is not a root of the original equation.

## Suggested 50-minute sequence

- **0–5 min:** Retrieve L05: what is a residual, and how do we verify a root?
- **5–12 min:** Introduce $x_{k+1}=g(x_k)$ and $x_s=g(x_s)$.
- **12–25 min:** Complete the three paper iteration tables and predict the behaviour.
- **25–38 min:** Use cobweb diagrams and the existing L06 notebook to test the predictions.
- **38–45 min:** Compare the maps and discuss starting points, local slope, and residual checks.
- **45–50 min:** Complete the exit question and connect to L07:
  
  $$
  x_{k+1}=x_k-\alpha f'(x_k)
  $$
  
  is also a fixed-point iteration.

## Exit question

> An iteration stops changing at $x=0$, but the original residual is $f(0)=-1$. Has the equation been solved? What should you check before trusting the result?

## Tool plan

- Paper calculations and sketches come first.
- Reuse the L06 fixed-point notebook for cobweb diagrams and residual checks.
- Open the notebook in **Edit code** so `student_map(x)` is visible for the seminar exercise.
- Display the selected residual and iteration equation above the cobweb plot.
- Keep the notebook controls simple: choose a map, change the starting value, and inspect the sequence and residual.
- Preserve a static fallback containing the three iteration tables and the expected convergence behaviours.

## Omit or defer

Do not include Newton convergence rates, numerical differentiation, or the full free-energy model in this seminar. The fixed-point idea is sufficient and gives students a framework for understanding gradient descent in L07.

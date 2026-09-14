# Lecture 5: finding a desired state by root finding

## Scope decision

Treat the latest instructor notes as authoritative. L05 introduces scalar root finding through the ideal-gas and van der Waals equations of state. It includes grid search, the Intermediate Value Theorem, bisection, regula falsi, and a brief view of SciPy's bracketed solvers. Secant and Newton methods remain in L06.

The class should emphasize why each algorithm works, what information it uses, and how to verify a reported root. Python syntax remains secondary.

## Driving materials question

At a specified temperature, pressure, and amount of gas, what volume does an equation of state predict?

## Learning outcomes

By the end of L05, students should be able to:

1. Given an equation of state and a target pressure, **formulate** a scalar residual $F(V)=P(V)-P_{\mathrm{target}}$ and state the physical domain of $V$. (Apply)
2. Given a graph or table of a continuous residual, **identify** a sign-changing bracket and explain the guarantee supplied by the Intermediate Value Theorem. (Analyze)
3. Given a valid bracket, **trace** bisection and regula falsi and report an estimate together with its bracket, residual, and stopping rule. (Apply)
4. **Compare** grid search, bisection, and regula falsi by function evaluations, guarantees, and likely failure modes, including the possibility of multiple roots. (Evaluate)

## Outcome alignment

| Outcome | Learning evidence | Teaching activity | Assessment form |
|---|---|---|---|
| Formulate $F(V)=0$ | Correct residual, units, and domain $V>nb$ | Rearrange ideal and van der Waals EOS | Short written formulation |
| Identify a bracket | Mark $a,b$ and justify $F(a)F(b)<0$ | Predict from EOS graph, then test values | Poll or paired explanation |
| Trace two methods | Two correct interval updates and a reported stopping check | Interactive method stepper | Annotated iteration table |
| Compare methods | Defensible method choice using evaluations and guarantees | Grid/bisection/regula falsi comparison | Exit response |

## Essential equations

Ideal gas:

$$
PV=nRT,\qquad P(V)=\frac{nRT}{V}.
$$

van der Waals equation for $n$ moles:

$$
\left(P+\frac{a n^2}{V^2}\right)(V-nb)=nRT,
$$

so

$$
P(V)=\frac{nRT}{V-nb}-\frac{a n^2}{V^2},
\qquad V>nb.
$$

For a specified pressure,

$$
F(V)=P(V)-P_{\mathrm{target}}.
$$

Bisection trial point:

$$
x_s=\frac{a+b}{2}.
$$

Regula falsi trial point:

$$
x_s=b-F(b)\frac{b-a}{F(b)-F(a)}
=\frac{aF(b)-bF(a)}{F(b)-F(a)}.
$$

After $k$ bisection updates, the bracket width is $(b_0-a_0)/2^k$. To make it no greater than $\tau_x$:

$$
k\geq\left\lceil\log_2\left(\frac{b_0-a_0}{\tau_x}\right)\right\rceil.
$$

This guarantees an interval-width tolerance. A residual tolerance requires a separate check because $|F(x)|$ also depends on the local slope and the units of $F$.

## Likely misconceptions

- Rearranging analytically and solving numerically are competing answers. They are two routes to the same physical condition; numerical inversion remains available when an explicit inverse is unavailable or inconvenient.
- $F(a)F(b)<0$ identifies exactly one root. It guarantees at least one root for a continuous function.
- Every root changes sign. A repeated or tangent root can touch zero without a sign change.
- A sign change always proves a root. A discontinuity can change sign without crossing zero; continuity must be checked.
- The grid point with the smallest $|F|$ must be a root. A grid search always returns a closest sampled point, even if no root exists in the interval.
- A small residual always means a small error in $x$. A flat function can have a small residual far from the root in $x$ units.
- Regula falsi is always faster than bisection. It can use function magnitudes effectively, but one endpoint can stagnate.
- A van der Waals root is automatically the observed equilibrium state. Multiple mathematical roots need stability and phase-equilibrium interpretation.

## 50-minute sequence

- **0–5 min:** Gas-compression teaser. Given $n,T,P$, predict how to obtain $V$ from the ideal-gas law.
- **5–11 min:** Compare explicit inversion with a graph of $P(V)$ and a horizontal target-pressure line. Define $F(V)=P(V)-P_{\mathrm{target}}$.
- **11–18 min:** Replace the ideal EOS with the van der Waals EOS. State $V>nb$. Students predict the number and approximate positions of intersections in the EOS explorer.
- **18–23 min:** Grid search. Connect grid spacing to cost and diagnose the no-root problem.
- **23–29 min:** Intermediate Value Theorem and sign-changing brackets. State continuity explicitly. Students choose a bracket.
- **29–39 min:** Trace bisection for two or three updates. Derive the bracket-width bound and distinguish $x$ tolerance from residual tolerance.
- **39–45 min:** Replace the midpoint by the regula falsi intercept. Compare its first steps with bisection and show a stagnation case if time permits.
- **45–48 min:** Show SciPy's bracketed `root_scalar` methods as trusted production tools after inspecting the transparent algorithms.
- **48–50 min:** Exit question on finding and interpreting all van der Waals roots.

## Demo plan

Use at most two embedded marimo demos.

1. **EOS and grid/bracket explorer:** ideal versus van der Waals model, gas parameters, reduced temperature and pressure, grid density, bracket endpoints, and optional reference roots. Students predict intersections before revealing roots.
2. **Bracket-method comparison:** bisection versus regula falsi on the selected bracket, with iteration history, function-evaluation count, bracket width, residual, and a SciPy comparison if this remains readable.

A single combined notebook is acceptable if the controls remain focused. The static fallback should include the EOS equations, one below-critical van der Waals curve with three intersections, and a short bisection/regula falsi comparison table.

## Credibility checks

A reported volume must include:

- units and physical domain $V>nb$;
- the initial sign-changing bracket;
- the final root estimate and residual in pressure units;
- interval width or change in estimate;
- iteration/function-evaluation count;
- physical interpretation when several roots exist.

## Exit / take-home question

For a subcritical van der Waals isotherm, choose a pressure for which the horizontal pressure line intersects the curve three times.

1. How would you systematically find all sign-changing brackets rather than only one root?
2. Why can one call to a bracketed solver return only one root?
3. What do the three volume roots mean physically, and which additional stability or phase-equilibrium argument is needed before calling one the observed state?

## Deferred material

- Derivation of cubic roots by formula.
- Secant and Newton methods (L06).
- Detailed convergence-order proofs.
- Maxwell equal-area construction and a full real-fluid phase-equilibrium treatment. Mention these only when interpreting the three-root regime.

## Instructor reflection after class

- Could students formulate the residual before seeing code?
- Did they state continuity when invoking the sign-change test?
- Could they distinguish an interval-width guarantee from a residual tolerance?
- Did the van der Waals example illuminate multiple roots, or did its thermodynamics overwhelm the numerical idea?
- Should L06 begin with regula-falsi stagnation as the motivation for secant and hybrid methods?

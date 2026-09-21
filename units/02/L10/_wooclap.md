### Q1 · Which numerical task?

**Matching · 90 seconds.** Match each question to its most direct numerical task. Use each task once: **root finding, minimization, interpolation, regression**.

| Physical or numerical question | Your choice |
|---|---|
| At fixed $T,P$, which volume lies at the bottom of a free-energy well? | |
| At fixed $T$, which pressure makes the two minimum free energies equal? | |
| Which curve passes through five calculated pressures to estimate between them? | |
| Which $C,D$ best represent the overall trend in $\ln P\approx C-D/T$? | |

### Q2 · What does the bracket establish?

**Single choice · 60 seconds.** Both free-energy minima exist throughout a pressure interval. Their difference $\Delta G(P)$ is continuous there and has opposite signs at the endpoints. What follows from this information?

A. There is exactly one coexistence pressure in the interval.  
B. There is at least one coexistence pressure in the interval.  
C. Newton–Raphson converges from every starting pressure in the interval.  
D. Every pressure in the interval gives two equally stable phases.

### Q3 · The volume has stopped changing

**Single choice · 60 seconds.** A fixed-point volume calculation gives a step of $10^{-8}$ L/mol and an EOS pressure residual of $2$ bar. The acceptance limits are $10^{-6}$ L/mol and $0.01$ bar, respectively. What should we report?

A. Accepted: the volume step is small enough.  
B. Accepted: printing fewer volume digits will remove the discrepancy.  
C. Not accepted: the pressure equation is still unsatisfied.  
D. No physical solution exists at these conditions.

### Q4 · Two successful minimizations

**Single choice · 75 seconds.** At one subcritical temperature and trial pressure, both volume minimizations converge and pass their local-minimum checks. They give

$$
G_{\mathrm{liq}}=-1200\ \mathrm{J/mol},\qquad
G_{\mathrm{gas}}=-1250\ \mathrm{J/mol}.
$$

Which interpretation is justified within this liquid–gas model?

A. The phases coexist because both minimizations converged.  
B. The liquid is favoured because its volume is smaller.  
C. The gas is favoured, and this pressure is not the coexistence pressure.  
D. One minimization must have failed because the energies differ.

### Q5 · Exact at five temperatures

**Single choice · 60 seconds.** An interpolator reproduces all five calculated saturation pressures. Which additional test most directly checks its accuracy between those temperatures?

A. Print the original pressures with more decimal places.  
B. Confirm again that the interpolator passes through the five points.  
C. Compare with a fresh coexistence solve at an intermediate temperature.  
D. Replace it with a higher-degree polynomial without calculating more points.

### Q6 · Same function form, different fitted coefficients

**Single choice · 75 seconds.** Two fits use the same positive pressure data and the form $P=\exp(C-D/T)$ bar. One minimizes squared pressure residuals, the other squared log-pressure residuals. Both numerical optimizations converge accurately. Why can they give different $C,D$?

A. Logarithms change how discrepancies contribute to the objective.  
B. One optimizer must have failed.  
C. Taking logarithms reduces the number of data points.  
D. They cannot differ if the numerical tolerance is small enough.

::: {.callout-note collapse="true"}
## Wooclap answer key and discussion prompts

Allow about seven minutes for the initial responses and eight for explanations and a revote where useful. Shuffle the matching choices in Wooclap.

| Question | Answer and reasoning | Misconceptions diagnosed | Follow-up |
|---|---|---|---|
| **1 · Apply** | Minimization, root finding, interpolation, regression, in that order. The outer root solver changes pressure, while the inner minimizers change volume. | Confusing the unknown of one calculation with another, or exact matching with fitting. | Which inputs stay fixed inside each numerical call? |
| **2 · Understand** | **B.** Continuity and a sign change establish at least one zero. | A assumes uniqueness from a sign change alone. C transfers the bracketing guarantee to an open method. D confuses two minima with equal minima. | Why must the supplied code keep both minima available throughout the bracket? |
| **3 · Analyze** | **C.** The pressure residual is 200 times its tolerance. | A confuses a small step with solving the equation. B confuses display precision with numerical accuracy. D turns an iteration failure into a claim of nonexistence. | If our coexistence residual is tiny but argon pressure differs from NIST by 34%, what should we investigate instead of reducing the solver tolerance? |
| **4 · Analyze** | **C.** The gas has lower free energy and $\Delta G=+50$ J/mol. | A confuses solver convergence with coexistence. B selects by volume rather than free energy. D assumes different local minima indicate an error. | What value must the outer pressure solver make zero? |
| **5 · Evaluate** | **C.** A new temperature tests a prediction where no interpolation condition was imposed. | A substitutes extra digits for evidence. B only rechecks the construction. D assumes more flexibility improves accuracy. | Does agreement with that fresh solve also establish agreement with experiment? |
| **6 · Analyze** | **A.** Log-pressure residuals measure pressure ratios, changing the objective compared with absolute pressure differences. | B treats a changed objective as solver failure. C confuses transformed values with fewer data. D assumes all objectives share a minimum. | Which residual is more appropriate if pressure uncertainty is roughly proportional to pressure? |

The follow-up to Q3 separates numerical verification from physical validation. The six questions revisit L05–L09 without requiring another demonstration of the helper functions.
:::

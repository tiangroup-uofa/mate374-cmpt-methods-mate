# A2 draft notes

The instructor's original `assignments/A2/idea.qmd` is unchanged. The handout follows A1's submission instructions, Problem/section numbering, parameter values, units, and unique-pair convention. Marks appear only on the four main problems (20/30/20/30).

## Completed in this pass

- Q1 sample answers retain multiple acceptable explanations.
- Q2 uses the symmetric regular-solution model without identifying a real alloy as exactly symmetric. H = Omega/(k_B T); Omega is the interaction parameter, while the mixing enthalpy per atom is Omega*x*(1-x).
- The exponential rearrangement selects a map that converges to the outer roots for H = 2.5. Starting values, stopping criterion, and update counting are explicit.
- Q3 uses eight reproducible, slightly noisy samples of the A1 argon LJ curve. No sample is at the exact root or minimum. Natural cubic boundary conditions make the answer reproducible. The fixed additive noise and rounding are in `check_answers.py`.
- Q4 supplies the residual definition and implementation, with the numerical check and fitted values deferred. RMSE is included because the total squared residual is not directly comparable across different N.
- Plain Python scaffolds are supplied for checking the exercise before building editable browser demos. The intended interactive Q2 plot and Q3/Q4 demos remain a later implementation task.

## Decisions for the quantum dataset

1. Choose the element, electronic-structure method, and energy reference. For a single-element cluster, the supplied target should be E_cluster - n*E_isolated_atom using consistent quantum settings. Argon requires a method that adequately describes dispersion.
2. Choose up to 50 fixed geometries with useful variation in distances and cluster sizes. Fix the ordering for nested 10/20/30/40/50 subsets. Keep fitted coordinates unchanged.
3. After inspecting the data, select sensible positive parameter bounds, initial values, and objective scaling/tolerances. Recompute the two-cluster checksum and all fitted results from the actual supplied data.
4. Decide whether to add held-out clusters. The current parity plot measures agreement on fitted data only. Parameter stability as N grows is an observation to investigate, not a promised result.

The checker tests the Q4 implementation against synthetic LJ totals only. These tests are not presented as DFT data or fitted quantum results.

## Local checks

```sh
uv run --locked python answer-keys/A2/check_answers.py
quarto render answer-keys/A2/answers.qmd --to pdf
```

For a handout-only render without website hooks, copy `assignments/A2/index.qmd` into a temporary directory outside the website project, run `quarto render <temporary-directory>/index.qmd --to pdf`, and copy `A2.pdf` back to `assignments/A2/`. No site render targets, hooks, navigation, or answer staging were changed.

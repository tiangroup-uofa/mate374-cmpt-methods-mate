# L08–L09: first implementation and next-round review

Author notes; not a lecture page. The two student-facing sources are `L08/index.qmd` and `L09/index.qmd`.

## Four questions for the next round

1. **How much polynomial algebra should students do by hand?** The draft keeps the two-point Lagrange construction visible and folds the three-point example, Newton divided-difference triangle, and straight-line least-squares derivation. Should the triangle and coefficient formulas become assessed hand calculations, or remain explanatory background?
2. **Where should the phase-diagram story end in these two lectures?** L08 has the working sparse-boundary demo and original free-energy check. L09 closes with an optional Clausius–Clapeyron fitting connection. Is that enough continuity, or should a later round add a fitted-boundary comparison to the same capstone?
3. **Which Arrhenius dataset do you want to teach with?** The current synthetic rate constants make the objective difference clear: the log-space fit gives 79.46 kJ/mol and the unweighted direct fit 69.07 kJ/mol for data generated with 80 kJ/mol and relative scatter. Keep this deliberately visible comparison, use diffusivity instead of rate, or replace it with a measured materials dataset?
4. **What should get the final ten minutes of L09?** The current route prioritizes the Arrhenius comparison, then introduces basis functions and matrix dimensions. Weighting, covariance, and alternative optimizers can stay as reading. Should the next round spend more class time on uncertainty, or on preparing students for systems of equations?

## Proposed 50-minute routes

These assume one 50-minute meeting per lecture. The expanded notes are a reference; the collapsed derivations are not an additional in-class topic list.

| Minutes | L08 | L09 |
|---|---|---|
| 0–5 | Retrieve L07: what calculations make one phase-boundary point? | Contrast preserving each point with estimating a noisy trend. |
| 5–13 | Zero-residual conditions; two-point Lagrange; idea of adding a Newton term. | Residuals, squared-error objective, two unknown line coefficients. |
| 13–20 | Runge prediction and point-count experiment. | Edit the tensile fit; interpret modulus and residuals. |
| 20–28 | Piecewise curves, spline conditions, and coefficient count. | Arrhenius transformation; identify what the slope means. |
| 28–36 | Edit NumPy/SciPy interpolation calls; test shape and out-of-range queries. | Compare log-space and direct fits; inspect both residual plots. |
| 36–46 | Sparse CO₂ boundary: held-out pressure, ΔG, timing, critical endpoint. | Basis functions; revisit polynomial degree; count matrix dimensions. |
| 46–50 | Explain why zero node residual is insufficient; bridge to noisy data. | Choose a Python function and explain why the best fit need not have zero error. |

If time is short, make Newton's triangle and quadratic-spline counting after-class reading in L08. In L09, keep weighting to one comparison and leave covariance and general optimizers collapsed. Preserve the Python experiments and final checks.

## Outcome alignment

| Lecture / outcome | Bloom level | In-class evidence | Follow-up check |
|---|---|---|---|
| L08: formulate coefficient residuals | Apply | Identify data versus coefficients in Fᵢ(a)=0. | Write the interpolation conditions for a short table. |
| L08: explain polynomial behaviour | Understand / Analyze | Predict and diagnose Runge oscillations. | Explain node residual versus between-point error. |
| L08: use interpolation calls | Apply | Edit query temperature, method, and extrapolation option. | Select a method for a monotone property table. |
| L08: assess the phase estimate | Evaluate | Compare with a direct solve and check ΔG. | Reject a coexistence extension above Tc. |
| L09: formulate the objective | Apply | Label known data, unknown line coefficients, and residuals. | Write SSE for a quadratic model. |
| L09: fit and interpret parameters | Apply | Run the tensile and Arrhenius calls with units. | Report modulus / activation energy and a residual measure. |
| L09: compare transformed and direct fits | Analyze | Check both SSE columns and both residual plots. | Explain why the objectives give different parameters. |
| L09: assess the model | Evaluate | Increase polynomial degree; compare numerical fit and physical usefulness. | Propose an independent prediction check. |

## Demos and fallbacks

- `activities/l08_lagrange.edit.py`: choose 2–7 points from a seven-point table, plot all selected $L_i(x)$ basis polynomials, and compare their interpolating curve.
- `activities/l08_runge.edit.py`: separate Runge interpolation and Gaussian extrapolation cases, point count and point placement, a data-range y-scale, and a switch for shaded extrapolated regions; the known functions permit genuine error checks.
- `activities/l08_scipy_interpolation.edit.py`: small editable property table and library calls; cubic overshoot versus PCHIP; explicit out-of-range rejection.
- `activities/l08_co2_interpolation.edit.py`: 4–10 computed coexistence states, three interpolation choices, timing, held-out check, and extrapolation past the critical point.
- `activities/l09_least_squares.edit.py`: editable tensile data and polynomial degree; coefficients, SSE, RMSE, R², and residual plot.
- `activities/l09_arrhenius.edit.py`: editable linearized and direct calls; original/log plots, both residuals, and one weighting control.

The notes provide prediction prompts and interpretation; notebooks keep prose short. Plotting/support cells are collapsed. All six have static figures and numerical/code fallbacks in the Quarto sources. Figures use labelled axes and legends in addition to colour.

Browser startup needs an initial Python/package download. If an embed fails, use its direct link or reset control; the printed/static figures provide the same discussion route. An invalid student edit can legitimately raise an error: restore the edited cell or reset rather than silently repairing the data. Full classroom browser interaction remains a final instructor smoke check; no prolonged browser debugging was done in this pass.

## Sources and preserved plans

- Original comments and draft text: `_archive/lectures-08-09-20260920-180708-planning/`, with `SHA256SUMS`.
- Kiusalaas (2013), Chapter 3, especially §§3.2–3.4, using the supplied `l08-09-kiusalaas-book-extract.txt`. The coefficient conditions, spline interpretation, and distinction between log-space and original-space residuals follow this source.
- Fall 2025 MATE 374 Lecture 11: straight-line least squares, slides 10–11 (the Sx/Sy/Sxx/Sxy formulas), slide 14 (transformation table), and general linear forms. The equation slides were checked visually in the PDF.
- Fall 2025 Lecture 12: polynomial regression, Lagrange, Newton, and divided differences; Lecture 13: continuation of the divided-difference construction. Multivariate interpolation is deferred because it is outside the user's two-lecture outline.
- L01–L07: question-led structure, measurable outcomes, prediction prompts, collapsed derivations, library calls, and the same dimensional vdW CO₂ model.

Corrections to inherited claims: degree is *at most* N−1; each Lagrange basis selects its own point; linear-spline slopes generally jump; quadratic and cubic splines need one and two additional conditions respectively; R² can be computed for nonlinear fits but needs careful interpretation; high polynomial degree has no universal safe cutoff. Fitting/extrapolation still needs physical validation.

## Local checks and build state

Passed:

- `uv run --locked marimo check activities/l08_*.edit.py activities/l09_*.edit.py`
- Non-interactive execution of all six notebooks.
- `uv run --locked python scripts/l08_l09_figures.py --check-only`: node residuals, Runge point layouts, spline endpoint conditions and monotonicity, the CO₂ equilibrium reference, straight-line normal conditions, nonlinear starts, relative weighting, and exact synthetic parameter recovery.
- Executed WASM export for all six notebooks, including the exporter's isolated scientific-Python environment.
- `quarto render --to html --no-clean`: exactly L08 and L09 rendered; a second render reused the six-notebook export cache.
- All ten displayed Python blocks executed with their stated input arrays; source links, iframe targets, exported auto-run settings, and archive checksums were checked.
- `quarto inspect`: two default inputs; the full profile retains the full render patterns, navigation including S03 and L09, full notebook export, figure generation, and answer-PDF hooks. The full site was not rendered.

**Keep the focus configuration until explicit push approval.** Daily `quarto preview` / `quarto render --to html` rebuilds only these lectures and their scripts. Other notebook exports are left untouched.

The complete pre-focus configuration is checkpointed in `_archive/quarto-checkpoints/20260920-182224-before-l08-l09-focus/`. At approval, restore its default render list and shared `_marimo_export.py` hook, retaining any later navigation edits. Remove the temporary duplicate full-profile render/export entries once the base configuration is restored. Do not blindly overwrite later edits with the checkpoint.

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
#     "scipy>=1.14",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.optimize import root_scalar

    return mo, np, root_scalar


@app.cell
def _(mo):
    mo.md(r"""
    # Try it: one materials calculation in Python, NumPy, and SciPy

    **Question.** A metal rod expands according to

    $$
    \Delta L = \alpha L_0 \Delta T.
    $$

    How much does it expand over several temperature changes, and what
    temperature change reaches a specified dimensional tolerance?

    Open **Edit code**. Change the ordinary values in the input cell,
    run that cell, and watch the later results update. You are not
    expected to memorize the code. Use the linked documentation in
    Appendix A1 when you do not recognize a function.
    """)
    return


@app.cell
def _(np):
    # Start here. Change one value, then run this cell again.
    length0_mm = 100.0
    alpha_per_K = 23.0e-6
    delta_T_values_K = np.array([-50.0, 0.0, 50.0, 100.0, 150.0])
    target_expansion_mm = 0.20
    return alpha_per_K, delta_T_values_K, length0_mm, target_expansion_mm


@app.cell
def _():
    def thermal_expansion(length_mm, alpha, delta_T_K):
        """Return length change in mm for a uniform temperature change."""
        return alpha * length_mm * delta_T_K

    return (thermal_expansion,)


@app.cell
def _(
    alpha_per_K,
    delta_T_values_K,
    length0_mm,
    np,
    thermal_expansion,
):
    expansion_values_mm = thermal_expansion(
        length0_mm,
        alpha_per_K,
        delta_T_values_K,
    )
    final_lengths_mm = length0_mm + expansion_values_mm
    largest_expansion_mm = np.max(expansion_values_mm)
    return expansion_values_mm, final_lengths_mm, largest_expansion_mm


@app.cell
def _(
    delta_T_values_K,
    expansion_values_mm,
    final_lengths_mm,
    largest_expansion_mm,
    mo,
):
    rows = "\n".join(
        f"| {delta_T:8.1f} | {change:10.4f} | {final:12.4f} |"
        for delta_T, change, final in zip(
            delta_T_values_K,
            expansion_values_mm,
            final_lengths_mm,
        )
    )
    mo.md(
        fr"""
        ## NumPy result

        One function evaluated an entire array of temperature changes.

        | $\Delta T$ (K) | $\Delta L$ (mm) | Final length (mm) |
        |---:|---:|---:|
        {rows}

        The largest expansion in this array is
        **{largest_expansion_mm:.4f} mm**.
        """
    )
    return


@app.cell
def _(
    alpha_per_K,
    length0_mm,
    root_scalar,
    target_expansion_mm,
    thermal_expansion,
):
    def tolerance_residual(delta_T_K):
        predicted_change_mm = thermal_expansion(
            length0_mm,
            alpha_per_K,
            delta_T_K,
        )
        return predicted_change_mm - target_expansion_mm

    solution = root_scalar(tolerance_residual, bracket=[0.0, 500.0])
    required_delta_T_K = solution.root
    final_residual_mm = tolerance_residual(required_delta_T_K)
    return final_residual_mm, required_delta_T_K, solution


@app.cell
def _(
    final_residual_mm,
    mo,
    required_delta_T_K,
    solution,
    target_expansion_mm,
):
    mo.callout(
        mo.md(
            fr"""
            ## SciPy result

            To reach an expansion of **{target_expansion_mm:.3f} mm**,
            the required temperature change is
            **{required_delta_T_K:.3f} K**.

            - solver converged: `{solution.converged}`
            - residual at the reported answer:
              `{final_residual_mm:.2e} mm`

            We did not write a root-finding package. We supplied SciPy
            with a function, a bracket, and a question. Later lectures
            will explain how this solver works and how to judge it.
            """
        ),
        kind="success" if solution.converged else "warn",
    )
    return


@app.cell
def _(alpha_per_K, length0_mm, mo, thermal_expansion):
    zero_change_check_mm = thermal_expansion(length0_mm, alpha_per_K, 0.0)
    mo.callout(
        mo.md(
            fr"""
            ## Quick check

            At $\Delta T=0$, the predicted expansion is
            **{zero_change_check_mm:.1e} mm**.

            This is a limiting-case check. The code ran, but this check
            gives us a reason to believe that the equation and units were
            entered consistently.
            """
        ),
        kind="success" if zero_change_check_mm == 0.0 else "danger",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Small changes to try

    Work through these in order. If something fails, read the last line
    of the error message and undo only your most recent change.

    1. Change `length0_mm` from `100.0` to `250.0`.
    2. Add `200.0` to `delta_T_values_K`.
    3. Change the target expansion from `0.20` to `0.40` mm.
    4. Find `np.min` in the NumPy reference and report the largest
       contraction in the array.
    5. Change the SciPy bracket from `[0.0, 500.0]` to `[0.0, 10.0]`.
       Read the error, explain why the bracket fails, and restore it.

    You have already used variables, a function, an array, a library
    function, a loop hidden inside a table display, and a SciPy solver.
    That is enough for the first week.
    """)
    return


if __name__ == "__main__":
    app.run()

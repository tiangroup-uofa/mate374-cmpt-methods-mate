# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numba>=0.61",
#     "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import time

    import marimo as mo
    import numba
    import numpy as np

    return mo, np, numba, time


@app.cell
def _(mo):
    mo.md(r"""
    # Optional native extension: JIT-compile the π loop

    This is the same Leibniz algorithm used in the local browser notebook,
    but `numba` compiles the loop with a native LLVM toolchain. That toolchain
    is not a dependable WASM dependency, so use the **Open in molab** link
    from the lecture page for this comparison.

    The first compiled call includes JIT compilation. The hot call reuses
    the compiled machine code. Compilation is an implementation detail: the
    numerical result must still agree with the transparent reference.
    """)
    return


@app.cell
def _(mo):
    terms = mo.ui.slider(
        start=10_000,
        stop=2_000_000,
        step=10_000,
        value=200_000,
        include_input=True,
        label="Number of terms, N",
        show_value=True,
    )
    run = mo.ui.run_button(label="Compile and benchmark")
    mo.hstack([terms, run], align="end", justify="start", gap=2)
    return run, terms


@app.cell
def _(numba):
    def pi_leibniz_loop(n_terms):
        total = 0.0
        for k in range(n_terms):
            sign = 1.0 if k % 2 == 0 else -1.0
            total += sign / (2.0 * k + 1.0)
        return 4.0 * total

    @numba.njit(cache=False)
    def pi_leibniz_jit(n_terms):
        total = 0.0
        for k in range(n_terms):
            sign = 1.0 if k % 2 == 0 else -1.0
            total += sign / (2.0 * k + 1.0)
        return 4.0 * total

    return pi_leibniz_jit, pi_leibniz_loop


@app.cell
def _(mo, np, pi_leibniz_jit, pi_leibniz_loop, run, terms, time):
    if not run.value:
        result = mo.callout(
            mo.md(
                "Press **Compile and benchmark** after choosing a term count. "
                "The first JIT call includes compilation."
            ),
            kind="info",
        )
    else:
        n_terms = int(terms.value)

        start = time.perf_counter()
        reference = pi_leibniz_loop(n_terms)
        python_seconds = time.perf_counter() - start

        start = time.perf_counter()
        first_jit = pi_leibniz_jit(n_terms)
        compile_seconds = time.perf_counter() - start

        start = time.perf_counter()
        hot_jit = pi_leibniz_jit(n_terms)
        hot_seconds = time.perf_counter() - start

        absolute_difference = abs(reference - hot_jit)
        speedup = python_seconds / hot_seconds if hot_seconds else np.inf
        result = mo.md(
            fr"""
            ## Native benchmark

            | Implementation | Wall time |
            |---|---:|
            | Python reference loop | {python_seconds:.5f} s |
            | First JIT call (compile + run) | {compile_seconds:.5f} s |
            | Reused JIT call | {hot_seconds:.5f} s |

            - hot-call speedup: **{speedup:.1f}×**
            - JIT estimate: `{first_jit:.15f}`
            - reference/JIT difference: **{absolute_difference:.3e}**

            The speedup is machine-dependent. Agreement with the reference is
            the verification check.
            """
        )
    result
    return


if __name__ == "__main__":
    app.run()

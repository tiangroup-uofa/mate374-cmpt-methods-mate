# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from time import perf_counter
    return mo, np, perf_counter, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## How much work does a dense spring-chain solve require?
    One fixed end, equal nearest-neighbor springs, and an end load give
    $u_i=iF/k$. **Predict:** how will solve time change when we double
    the number of moving atoms? Compare explicit inversion with `solve`.
    Press **Run dense benchmark** to time 512, 1,024, and 2,048 unknowns.
    The larger calculations may take a while in the browser. We will fit
    the measured times and extrapolate to one million unknowns without
    allocating a matrix of that size.
    """)
    return


@app.cell
def _(np):
    sizes = np.array([512, 1024, 2048])
    k = 5.0  # eV/Å²
    F = 0.001  # eV/Å
    repeats = 3
    return F, k, repeats, sizes


@app.cell
def _(mo):
    run_benchmark = mo.ui.run_button(label="Run dense benchmark")
    run_benchmark
    return (run_benchmark,)


@app.cell
def _(F, k, mo, np, perf_counter, repeats, run_benchmark, sizes):
    mo.stop(not run_benchmark.value, mo.md("Press **Run dense benchmark** to collect timings."))
    mo.stop(
        sizes.ndim != 1 or len(sizes) < 3
        or not np.issubdtype(sizes.dtype, np.integer)
        or np.any(sizes < 2) or np.any(sizes > 2048)
        or len(np.unique(sizes)) != len(sizes),
        mo.md("Choose at least three distinct integer sizes between 2 and **2,048**."),
    )
    mo.stop(not isinstance(repeats, int) or not 1 <= repeats <= 5,
            mo.md("Choose an integer repeat count from 1 to 5."))
    times_inverse = []
    times_solve = []
    for _N in mo.status.progress_bar(sizes, title="Benchmarking dense matrices"):
        _K = np.diag(np.full(_N, 2 * k))
        _K += np.diag(np.full(_N - 1, -k), 1)
        _K += np.diag(np.full(_N - 1, -k), -1)
        _K[-1, -1] = k
        _f = np.zeros(_N)
        _f[-1] = F
        _reference = np.arange(1, _N + 1) * F / k
        # Warm both library paths before collecting timings.
        np.linalg.solve(_K, _f)
        np.linalg.inv(_K) @ _f
        for _method, _results in [("inverse", times_inverse), ("solve", times_solve)]:
            _samples = []
            for _ in range(repeats):
                _start = perf_counter()
                if _method == "inverse":
                    _u = np.linalg.inv(_K) @ _f
                else:
                    _u = np.linalg.solve(_K, _f)
                _samples.append(perf_counter() - _start)
            _results.append(float(np.median(_samples)))
            _relative_error = np.linalg.norm(_u - _reference) / np.linalg.norm(_reference)
            _residual = np.linalg.norm(_K @ _u - _f) / np.linalg.norm(_f)
            print(f"N={_N:4d} {_method:7s}: {_results[-1]:.4g} s; "
                  f"relative error={_relative_error:.2e}; relative residual={_residual:.2e}")
    return times_inverse, times_solve


@app.cell
def _(mo, np, plt, sizes, times_inverse, times_solve):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _rows = []
    _target = 1_000_000
    for _label, _times in [("Inverse @ f", times_inverse), ("solve", times_solve)]:
        _line, = ax.loglog(sizes, _times, "o", label=f"{_label}: measured")
        # log(t) = log(c) + p log(N): a degree-one polynomial fit.
        _p, _log_c = np.polyfit(np.log(sizes), np.log(_times), 1)
        _grid = np.geomspace(min(sizes), max(sizes), 100)
        ax.loglog(_grid, np.exp(_log_c + _p * np.log(_grid)),
                  "--", color=_line.get_color(), label=f"{_label}: p = {_p:.2f}")
        _seconds = float(np.exp(_log_c + _p * np.log(_target)))
        _rows.append({
            "Method": _label,
            "Fitted exponent p": round(float(_p), 3),
            "Estimated seconds at N = 1,000,000": f"{_seconds:.3g}",
            "Estimated days": f"{_seconds / 86400:.3g}",
            "Estimated years (365 days)": f"{_seconds / (365 * 86400):.3g}",
        })
    ax.set(xlabel="Moving atoms N", ylabel="Median elapsed time (s)")
    ax.legend()
    fig.tight_layout()
    mo.vstack([
        fig,
        mo.md("### Power-law extrapolation to one million unknowns"),
        mo.ui.table(_rows, selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The fit is $t=cN^p$, obtained with `np.polyfit(log(N), log(t), 1)`.
    **How close is each exponent to 3?** Both methods factor the dense
    matrix, which has cubic leading-order work. `solve` then computes one
    displacement vector; inversion computes all columns of the inverse.
    The extra work can change the timing substantially while leaving the
    scaling exponent unchanged.

    **The million-unknown times are extrapolations**, far beyond our measured
    range. Browser overhead, memory traffic, and library behavior affect
    the fitted exponent. A dense matrix at that size needs **8 TB** before
    inverse storage or solver workspace, so memory already prevents this
    browser calculation. The next demo uses banded storage to change both
    the storage requirement and the computational scaling.
    """)
    return


if __name__ == "__main__":
    app.run()

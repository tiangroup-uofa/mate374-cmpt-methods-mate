# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
#     "matplotlib>=3.9",
#     "numba>=0.60.0",
#     "torch>=2.2.0",
# ]
# ///
"""Molab Full Environment Benchmark: Numba JIT, Parallelism, and GPU PyTorch.
Open directly in Molab with GPU runtime enabled.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    import math

    try:
        import numba
        has_numba = True
    except ImportError:
        has_numba = False

    try:
        import torch
        has_torch = True
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    except ImportError:
        has_torch = False
        device = "cpu"

    return device, has_numba, has_torch, math, mo, numba, np, plt, time, torch


@app.cell(hide_code=True)
def _(device, has_numba, has_torch, mo):
    mo.md(
        f"""
        # L04 Live Benchmark: Accelerating Buffon's Needle

        - **Numba JIT available:** `{has_numba}`
        - **PyTorch accelerator available:** `{has_torch}` (device: **`{device}`**)

        When run on **Molab** with GPU runtime enabled, this notebook runs genuine JIT machine compilation and CUDA kernels.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    problem_size_dropdown = mo.ui.dropdown(
        options={"10^5": 100_000, "10^6": 1_000_000, "10^7": 10_000_000, "10^8": 100_000_000},
        value="10^6",
        label="Problem size N",
    )
    run_btn = mo.ui.button(label="Run Benchmark Suite", value=0)
    mo.hstack([problem_size_dropdown, run_btn], justify="start", align="center")
    return problem_size_dropdown, run_btn


@app.cell(hide_code=True)
def _(has_numba, numba, np):
    if has_numba:
        @numba.njit(fastmath=True)
        def buffon_numba_single(N, seed):
            np.random.seed(seed)
            crossings = 0
            for _ in range(N):
                d = 0.5 * np.random.random()
                theta = 0.5 * np.pi * np.random.random()
                if d <= 0.5 * np.sin(theta):
                    crossings += 1
            return crossings

        @numba.njit(parallel=True, fastmath=True)
        def buffon_numba_parallel(N, n_threads, base_seed):
            chunk = N // n_threads
            crossings = 0
            # prange distributes across CPU workers
            for t in numba.prange(n_threads):
                np.random.seed(base_seed + t)
                local_c = 0
                for _ in range(chunk):
                    d = 0.5 * np.random.random()
                    theta = 0.5 * np.pi * np.random.random()
                    if d <= 0.5 * np.sin(theta):
                        local_c += 1
                crossings += local_c
            return crossings
    else:
        buffon_numba_single = None
        buffon_numba_parallel = None

    return buffon_numba_parallel, buffon_numba_single


@app.cell(hide_code=True)
def _(device, has_torch, torch):
    if has_torch:
        def buffon_pytorch_gpu(N, batch_size=5_000_000):
            remaining = N
            total_crossings = 0
            dev = torch.device(device)
            while remaining > 0:
                cur = min(remaining, batch_size)
                d = 0.5 * torch.rand(cur, device=dev)
                theta = 0.5 * torch.pi * torch.rand(cur, device=dev)
                crossings = (d <= 0.5 * torch.sin(theta)).sum().item()
                total_crossings += crossings
                remaining -= cur
            if dev.type == "cuda":
                torch.cuda.synchronize()
            return total_crossings
    else:
        buffon_pytorch_gpu = None

    return (buffon_pytorch_gpu,)


@app.cell(hide_code=True)
def _(
    buffon_numba_parallel,
    buffon_numba_single,
    buffon_pytorch_gpu,
    has_numba,
    has_torch,
    np,
    problem_size_dropdown,
    run_btn,
    time,
):
    _ = run_btn.value
    N = int(problem_size_dropdown.value)

    results = {}

    # 1. Pure Python (only up to 10^6)
    if N <= 1_000_000:
        import random, math
        t0 = time.perf_counter()
        c = 0
        for _ in range(N):
            d = 0.5 * random.random()
            th = 0.5 * math.pi * random.random()
            if d <= 0.5 * math.sin(th):
                c += 1
        t_py = time.perf_counter() - t0
        results["Pure Python"] = t_py

    # 2. Batched NumPy
    t0 = time.perf_counter()
    rng = np.random.default_rng(374)
    rem = N
    c_np = 0
    batch = min(N, 1_000_000)
    while rem > 0:
        cur = min(rem, batch)
        d = 0.5 * rng.random(cur)
        th = 0.5 * np.pi * rng.random(cur)
        c_np += int(np.count_nonzero(d <= 0.5 * np.sin(th)))
        rem -= cur
    t_np = time.perf_counter() - t0
    results["NumPy (batched)"] = t_np

    # 3. Numba JIT Single Core (warmup call first)
    if has_numba:
        buffon_numba_single(100, 42) # warmup
        t0 = time.perf_counter()
        buffon_numba_single(N, 374)
        t_numba = time.perf_counter() - t0
        results["Numba JIT (1 core)"] = t_numba

        # 4. Numba Parallel
        buffon_numba_parallel(100, 4, 42) # warmup
        t0 = time.perf_counter()
        buffon_numba_parallel(N, 8, 374)
        t_par = time.perf_counter() - t0
        results["Numba parallel (multi-core)"] = t_par

    # 5. PyTorch GPU
    if has_torch and buffon_pytorch_gpu is not None:
        buffon_pytorch_gpu(100) # warmup
        t0 = time.perf_counter()
        buffon_pytorch_gpu(N)
        t_gpu = time.perf_counter() - t0
        results["PyTorch GPU"] = t_gpu

    return N, results


@app.cell(hide_code=True)
def _(N, mo, plt, results):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    names = list(results.keys())
    times = list(results.values())
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#386cb0"][:len(names)]

    bars = ax.barh(names, times, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (seconds, log scale)")
    ax.set_title(f"Measured timing for N = {N:,} throws", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", which="both", alpha=0.25)

    for bar, t in zip(bars, times):
        ax.text(t * 1.15, bar.get_y() + bar.get_height() / 2, f"{t:.4f} s", va="center", fontsize=9)

    ax.set_xlim(min(times) * 0.5, max(times) * 5.0)
    fig.tight_layout()

    rows = "\n".join([f"- **{k}**: `{v:.5f}` seconds" for k, v in results.items()])
    summary = mo.md(
        f"""
        ### Measured Results ($N = {N:,}$):
        {rows}
        """
    )
    mo.vstack([fig, summary])
    return


if __name__ == "__main__":
    app.run()

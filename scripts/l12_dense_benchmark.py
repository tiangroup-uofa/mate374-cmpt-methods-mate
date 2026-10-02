"""Time dense inverse and solve for the L12 spring chain and save the data.

Run from the repository root: uv run --locked python scripts/l12_dense_benchmark.py
This is a benchmark, not part of render_figures.py. Rerunning it replaces
data/L12-dense-benchmark.json with timings from the current machine.
"""
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIZES = [512, 1024, 2048]
REPEATS = 5
K_SPRING, FORCE = 5.0, 0.001


def chain_matrix(n):
    K = np.diag(np.full(n, 2 * K_SPRING))
    K += np.diag(np.full(n - 1, -K_SPRING), 1)
    K += np.diag(np.full(n - 1, -K_SPRING), -1)
    K[-1, -1] = K_SPRING
    return K


def main():
    times = {"inverse": [], "solve": []}
    for n in SIZES:
        K = chain_matrix(n)
        f = np.zeros(n)
        f[-1] = FORCE
        np.linalg.solve(K, f)
        np.linalg.inv(K) @ f
        for method, run in [("inverse", lambda: np.linalg.inv(K) @ f),
                            ("solve", lambda: np.linalg.solve(K, f))]:
            samples = []
            for _ in range(REPEATS):
                start = perf_counter()
                run()
                samples.append(perf_counter() - start)
            times[method].append(float(np.median(samples)))
    data = {
        "sizes": SIZES,
        "median_seconds": times,
        "machine": f"{platform.machine()}, {os.cpu_count()} cores, NumPy {np.__version__}",
    }
    path = ROOT / "data" / "L12-dense-benchmark.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

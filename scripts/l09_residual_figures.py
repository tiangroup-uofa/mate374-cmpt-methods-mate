"""Generate the L09 degree 1, 3, and 10 residual comparison at 300 dpi.

Run: uv run --locked python scripts/l09_residual_figures.py
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "activities" / "l09_residual_patterns.edit.py"
    spec = spec_from_file_location("residual_patterns", source)
    notebook = module_from_spec(spec)
    spec.loader.exec_module(notebook)
    _, values = notebook.app.run()
    x, y = values["x"], values["y"]
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    sums = []
    for column, degree in enumerate((1, 3, 10)):
        model = Polynomial.fit(x, y, degree)
        residual = y - model(x)
        sse = np.sum(residual**2)
        coefficients = model.convert().coef
        sums.append(sse)
        print(f"Degree {degree} coefficients: {np.array2string(coefficients, precision=4)}")
        np.testing.assert_allclose(residual.sum(), 0, atol=1e-10)
        values["draw_fit"](axes[:, column], x, y, model, residual, degree)
        axes[0, column].set_title(f"Degree {degree} · SSE = {sse:.2f}")
        print(f"Degree {degree}: SSE = {sse:.6f}")
    assert sums[1] < 0.1*sums[0]  # Cubic terms explain the dominant pattern.
    assert sums[2] <= sums[1]  # Additional terms cannot increase training SSE.
    path = ROOT / "assets" / "L09-residual-patterns.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

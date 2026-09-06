# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib", "mpmath"]
# ///
"""Run with: uv run scripts/polygon_precision.py

Compare the cancellation-prone L02 recurrence. Binary128 is simulated
with mpmath at 113 significant bits, not claimed as a native NumPy type.
Its different exponent range is irrelevant for this experiment.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


def polygon_history(cast, sqrt):
    n, side = 6, cast(1)
    rows = []
    for doubling in range(70):
        estimate = cast(n // 2) * side
        # Convert the stored value exactly, without decimal display rounding.
        with mp.workprec(200):
            if isinstance(estimate, np.floating):
                numerator, denominator = float(estimate).as_integer_ratio()
                exact = mp.mpf(numerator) / denominator
            else:
                exact = mp.mpf(estimate)
            error = float(abs(exact - mp.pi))
        rows.append((doubling, n, error, side == 0))
        if side == 0:
            break
        # All operands and intermediate operations use the selected precision.
        side = sqrt(cast(2) - sqrt(cast(4) - side * side))
        n *= 2
    return rows


def main():
    histories = {}
    for dtype in [np.float16, np.float32, np.float64]:
        histories[f"{dtype.__name__} ({np.finfo(dtype).nmant + 1} bits)"] = (
            polygon_history(dtype, np.sqrt)
        )
    with mp.workprec(113):
        histories["binary128 precision (113 bits; simulated)"] = polygon_history(
            mp.mpf, mp.sqrt
        )

    print("Format | best-error doubling | sides | minimum absolute error | first zero-side doubling")
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, rows in histories.items():
        best = min(rows, key=lambda row: row[2])
        zero = next((row[0] for row in rows if row[3]), None)
        print(f"{label} | {best[0]} | {best[1]} | {best[2]:.6g} | {zero}")
        ax.semilogy([r[0] for r in rows], [r[2] for r in rows], ".-", label=label)
    ax.set(xlabel="Number of side doublings, k (N = 6 × 2ᵏ)",
           ylabel="Absolute error in π estimate",
           title="Higher precision delays cancellation in the polygon recurrence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    output = Path(__file__).resolve().parents[1] / "assets/L03-polygon-precision.png"
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

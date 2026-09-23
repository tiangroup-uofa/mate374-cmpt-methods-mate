"""Generate the L09 bitumen illustration as a 300-dpi PNG.

Run: uv run --locked python scripts/l09_intro_figures.py
These are synthetic teaching data, not measurements from the linked paper.
The diagnostics scaffold is retained below but is NOT regenerated: the
instructor has finalized its SVG/PNG by hand.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, ORANGE, GREEN = "#246b9e", "#c25b24", "#29805c"


def save(fig, name):
    fig.savefig(ROOT / "assets" / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"assets/{name}")


def viscosity():
    rng = np.random.default_rng(37409)
    # Four interleaved series, each spanning the temperature range. Distinct
    # inputs let one cubic interpolant pass through all 64 measurements.
    temperature = np.linspace(298, 368, 64)
    batch = np.arange(temperature.size) % 4
    offsets = np.array([-6.0, -2.0, 2.0, 6.0])[batch]
    scales = np.array([0.97, 1.02, 0.98, 1.03])[batch]

    # Scaled form avoids fitting an enormous alpha directly:
    # mu = A*(T/298)^(-p) = alpha*T^(-p), alpha = A*298^p.
    def power(T, A, p):
        return A * (T / 298.0) ** (-p)

    measured = power(temperature + offsets, 100.0, 8.0) * scales
    measured *= 1 + rng.normal(0, 0.12, temperature.size)
    parameters, _ = curve_fit(power, temperature, measured,
                              p0=[100, 8], bounds=(0, np.inf))
    assert 7 < parameters[1] < 9
    assert np.all(measured > 0)
    grid = np.linspace(298, 368, 2000)
    cubic = CubicSpline(temperature, measured)
    np.testing.assert_allclose(cubic(temperature), measured)
    assert np.all(np.isfinite(cubic(grid)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True,
                             layout="constrained")
    titles = ["Synthetic measurements", "Cubic-spline interpolation",
              "Power-law curve fitting"]
    colors = [BLUE, "#7b62a3", "#b88b25", "#578c90"]
    markers = ["o", "s", "^", "D"]
    upper = max(measured.max(), cubic(grid).max()) * 1.10
    lower = min(0, cubic(grid).min() * 1.10)
    for ax, title in zip(axes, titles):
        for series, (color, marker) in enumerate(zip(colors, markers)):
            selected = batch == series
            ax.scatter(temperature[selected], measured[selected], color=color,
                       marker=marker, s=20, zorder=3, label=f"Series {series+1}")
        ax.set(title=title, xlabel="Temperature, T (K)", ylim=(lower, upper))
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].set_ylabel("Viscosity, μ (Pa·s)")
    axes[1].plot(grid, cubic(grid), color=ORANGE, linewidth=1.4)
    axes[2].plot(grid, power(grid, *parameters), color=GREEN, linewidth=2.2)
    axes[2].text(0.05, 0.92, "μ = α T⁻ᵖ\n"
                 f"p = {parameters[1]:.2f}\n"
                 f"μ(298 K) = {parameters[0]:.1f} Pa·s",
                 transform=axes[2].transAxes, va="top", ha="left")
    # Keep the model annotation away from the low-temperature data.
    axes[2].texts[0].set_position((0.43, 0.92))
    save(fig, "L09-viscosity-fitting.png")
    print(f"Power fit: A={parameters[0]:.4f} Pa·s, p={parameters[1]:.4f}")


def diagnostics():
    rng = np.random.default_rng(109)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.linspace(0, 1, 12)
    grid = np.linspace(0, 1, 800)
    y_line = 0.8 + 1.5*x + rng.normal(0, 0.12, len(x))
    best = Polynomial.fit(x, y_line, 1)
    axes[0].scatter(x, y_line, color=BLUE, s=32, zorder=3)
    axes[0].plot(grid, 0.3 + 0.7*grid, "--", color=ORANGE,
                 label="Parameters not yet optimized", linewidth=2)
    axes[0].plot(grid, best(grid), color=GREEN,
                 label="Best-fitting line", linewidth=2)
    axes[0].set_title("1. Has the fitting procedure converged?")
    assert np.sum((y_line-best(x))**2) < np.sum((y_line-(0.3+0.7*x))**2)

    y = 0.7 + 3.5*(x-0.45)**2 + rng.normal(0, 0.10, len(x))
    axes[1].scatter(x, y, color=BLUE, s=32, zorder=3)
    for degree, color, style, label in [
        (0, ORANGE, "--", "Constant: underfitting"),
        (2, GREEN, "-", "Quadratic: captures the trend"),
        (11, "#8e579c", "-", "Degree 11: overfitting"),
    ]:
        model = Polynomial.fit(x, y, degree)
        axes[1].plot(grid, model(grid), style, color=color,
                     linewidth=1.8, label=label)
        if degree == 11:
            np.testing.assert_allclose(model(x), y, atol=1e-7)
    axes[1].set_title("2. Does the model have suitable flexibility?")
    for ax in axes:
        ax.set(xlabel="Input, x", ylabel="Measured response, y")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  frameon=False, fontsize=10)
    save(fig, "L09-fitting-diagnostics.svg")


if __name__ == "__main__":
    viscosity()
    # Do not regenerate the instructor's finalized diagnostics SVG/PNG.

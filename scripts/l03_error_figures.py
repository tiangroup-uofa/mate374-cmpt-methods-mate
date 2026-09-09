# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Generate L03 schematics: uv run scripts/l03_error_figures.py.

Patriot source: GAO/IMTEC-92-26 (1992), with the conversion explained at
https://www-users.cse.umn.edu/~arnold/disasters/patriot.html
The fixed-point reconstruction retains 23 fractional bits in a 24-bit word.
The error-balance plot is a qualitative log-log sketch, not measured errors.
"""
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"
BLUE, ORANGE, GREEN = "#28618a", "#bd5328", "#34744a"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def error_balance():
    # Work in log coordinates: arbitrary scales, deliberately unequal slopes.
    # There are no numerical ticks because this is not a quantitative model.
    log_effort = np.linspace(0, 6, 500)
    log_truncation = 3.8 - 0.85 * log_effort
    log_roundoff = -0.5 + 0.48 * log_effort
    log_total = np.log10(10**log_truncation + 10**log_roundoff)
    best = np.argmin(log_total)
    np.random.seed(374)  # Keep any sketch randomness reproducible.
    with plt.xkcd(scale=1.2, length=110, randomness=2):
        # Use an installed handwriting font when available; no font download needed.
        from matplotlib import font_manager
        available = {font.name for font in font_manager.fontManager.ttflist}
        handwriting = [name for name in
                       ["xkcd", "Humor Sans", "Comic Sans MS", "Chalkboard", "Chalkboard SE"]
                       if name in available]
        plt.rcParams["font.family"] = handwriting + ["DejaVu Sans"]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        fig.subplots_adjust(left=.12, bottom=.22, right=.96, top=.96)
        # Colour the two dominant-error branches, without extra extrapolated lines.
        ax.plot(log_effort[:best + 1], log_total[:best + 1], color=BLUE, lw=2.5)
        ax.plot(log_effort[best:], log_total[best:], color=ORANGE, lw=2.5)
        ax.text(.25, 3.9, "Truncation error", color=BLUE, fontsize=17)
        ax.text(4.1, 2.55, "Round-off error", color=ORANGE, fontsize=17)
        ax.scatter(log_effort[best], log_total[best], color="#333333", s=55, zorder=5)
        ax.annotate("diminished return", (log_effort[best], log_total[best]),
                    xytext=(2.0, 3.2), fontsize=17,
                    arrowprops={"arrowstyle": "->", "color": "#333333"})
        ax.set(xlim=(-.1, 6.2), ylim=(-.2, 4.5), xticks=[], yticks=[],
               xlabel="Increasing computational steps / effort (log scale)",
               ylabel="Absolute error (log scale)")
        # Arrow along the horizontal axis makes the direction explicit.
        ax.annotate("", xy=(1.015, 0), xytext=(.97, 0),
                    xycoords="axes fraction", annotation_clip=False,
                    arrowprops={"arrowstyle": "->", "color": "black"})
        fig.savefig(ASSETS / "L03-error-balance.png", dpi=180)
        plt.close(fig)


def patriot():
    exact_tick = Fraction(1, 10)
    stored_tick = Fraction(int(exact_tick * 2**23), 2**23)
    bias = exact_tick - stored_tick
    ticks = 100 * 3600 * 10
    time_error = float(ticks * bias)
    speed = 1700
    distance = speed * time_error
    assert abs(time_error - 0.34332275390625) < 1e-14
    assert 580 < distance < 590
    fig = plt.figure(figsize=(11, 5.4))
    fig.suptitle("Dhahran, 1991: a small time-conversion bias, a missed radar track",
                 fontsize=15, y=.97)
    boxes = [(.18, "INTEGER CLOCK", "100 hours of operation\n3,600,000 ticks"),
             (.50, "FINITE-PRECISION CONVERSION", "Seconds = ticks × stored 0.1\nBias ≈ 0.0000000954 s per tick"),
             (.82, "MAGNIFIED TIME ERROR", "3,600,000 × bias\n≈ 0.34 s")]
    for x, title, text in boxes:
        fig.text(x, .81, title, ha="center", color=BLUE, fontsize=10, weight="bold")
        fig.text(x, .71, text, ha="center", va="center", fontsize=12,
                 bbox={"boxstyle": "round,pad=.7", "fc": "#edf3f7", "ec": "#c6d4df"})
    fig.text(.34, .71, "→", ha="center", fontsize=22, color=BLUE)
    fig.text(.66, .71, "→", ha="center", fontsize=22, color=BLUE)
    ax = fig.add_axes([.07, .15, .86, .40])
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    ax.add_patch(Rectangle((1.3, .9), 2.6, 1.3, facecolor="#edf3f7",
                           edgecolor=BLUE, linewidth=2, linestyle="--"))
    ax.text(2.6, 1.55, "Predicted radar\nsearch window", ha="center", va="center", color=BLUE)
    ax.scatter([7.5], [1.55], s=170, marker=">", color=ORANGE)
    ax.text(7.5, 2.1, "Actual target", ha="center", color=ORANGE, weight="bold")
    ax.annotate("", (8.8, 1.55), (7.85, 1.55),
                arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2})
    ax.annotate("", (7.5, .6), (2.6, .6),
                arrowprops={"arrowstyle": "<->", "color": "#555555"})
    ax.text(5.05, .08, "Target travel: 1700 m/s × 0.34 s ≈ 580 m", ha="center", fontsize=13)
    fig.text(.5, .045, "Fixed-point conversion, not Python binary64 • Schematic, not to scale • Sources: GAO/IMTEC-92-26; Arnold (2000)",
             ha="center", fontsize=10, color="#555555")
    fig.savefig(ASSETS / "L03-patriot-timing.png", dpi=180)
    plt.close(fig)
    print(f"Patriot reconstruction: bias={float(bias):.12g} s/tick; "
          f"100 h error={time_error:.9f} s; target travel={distance:.3f} m")


if __name__ == "__main__":
    error_balance()
    patriot()

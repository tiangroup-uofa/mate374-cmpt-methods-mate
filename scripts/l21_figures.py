"""Draw the L21 figures: schematics of the MD engine and results from the completed notebook.

    uv run --locked python scripts/l21_figures.py [--schematics-only]
"""
import argparse
from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Ellipse, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DPI = 300
INK, RED, BLUE, GREEN, ORANGE, GREY = "#231f20", "#b5473a", "#3b6ea8", "#4a9a5b", "#e69f00", "#8a8f98"
PALE = "#f3f1ec"
MONO = "DejaVu Sans Mono"
STYLE = {"font.size": 13, "font.family": "Arial", "mathtext.fontset": "custom",
         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
         "svg.fonttype": "none"}


def canvas(w, h, scale=0.6):
    fig = plt.figure(figsize=(w * scale, h * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(0, h), aspect="equal")
    ax.axis("off")
    return fig, ax


def save(fig, name, svg=True, close=True):
    for ext in ("png", "svg") if svg else ("png",):
        fig.savefig(ASSETS / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    if close:
        plt.close(fig)
    print(f"Saved {name}")


def box(ax, x, y, w, h, ec=INK, fc="white", lw=1.6, r=0.2, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, ls=ls, zorder=z))


def arrow(ax, p, q, color=INK, lw=1.6, rad=0.0, style="-|>", ms=14, z=3, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def atom(ax, x, y, r=0.22, color=GREY, label=None, ls="-", fc=None, fs=11):
    ax.add_patch(Circle((x, y), r, fc=fc or color, ec=INK, lw=1.2, ls=ls, zorder=4))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=5,
                color="white" if fc is None and color in (RED, BLUE, GREEN, INK) else INK)


def table(ax, x, y, rows, cols, cw=0.5, ch=0.38, marks=None, lw=1.2):
    """Draw a rows x cols array with its top-left corner at (x, y); marks maps (row, col) to a colour."""
    marks = marks or {}
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x + c * cw, y - (r + 1) * ch), cw, ch, fc=marks.get((r, c), "white"),
                                   ec=INK, lw=lw, zorder=2))
    return [y - (r + 0.5) * ch for r in range(rows)], [x + (c + 0.5) * cw for c in range(cols)]


def _brace_profile(n=120):
    """Half of a curly brace: (u, w) with u from end (0) to tip (1) and w the outward offset in [0, 1]."""
    v = np.linspace(0, 1, n)
    end = 0.5 * np.sin(0.5 * np.pi * v)
    tip = 0.5 + 0.5 * (1 - np.cos(0.5 * np.pi * v))
    u = np.r_[0.25 * v, 0.25 + 0.5 * v, 0.75 + 0.25 * v]
    w = np.r_[end, np.full(n, 0.5), tip]
    return u, w


def brace(ax, x, y0, y1, width=0.25, color=INK, lw=1.5, left=False):
    """Vertical curly brace from y0 to y1 at x, pointing right (default) or left."""
    s = -1 if left else 1
    u, w = _brace_profile()
    ym = (y0 + y1) / 2
    for a in (y0, y1):
        ax.plot(x + s * width * w, a + (ym - a) * u, color=color, lw=lw, solid_capstyle="round")


def hbrace(ax, x0, x1, y, height=0.22, color=INK, lw=1.4):
    """Horizontal curly brace above a span, pointing up."""
    u, w = _brace_profile()
    xm = (x0 + x1) / 2
    for a in (x0, x1):
        ax.plot(a + (xm - a) * u, y + height * w, color=color, lw=lw, solid_capstyle="round")


def vdots(ax, x, y, gap=0.11, size=2.2):
    ax.plot([x] * 3, [y - gap, y, y + gap], "o", ms=size, color=INK)


def gear(ax, x, y, r, teeth=8, color=GREY):
    angles = np.linspace(0, 2 * np.pi, teeth * 4, endpoint=False)
    radii = np.tile([r, r, 1.28 * r, 1.28 * r], teeth)
    pts = np.c_[x + radii * np.cos(angles + np.pi / (teeth * 4)), y + radii * np.sin(angles + np.pi / (teeth * 4))]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=INK, lw=1.1, zorder=4))
    ax.add_patch(Circle((x, y), 0.38 * r, fc="white", ec=INK, lw=1.1, zorder=5))


def mini_axes(ax, x, y, w, h, xlabel, ylabel):
    arrow(ax, (x, y), (x + w, y), lw=1.3, ms=10)
    arrow(ax, (x, y), (x, y + h), lw=1.3, ms=10)
    ax.text(x + w, y - 0.12, xlabel, ha="right", va="top", fontsize=12)
    ax.text(x + 0.1, y + h, ylabel, ha="left", va="top", fontsize=12)


def md_workflow():
    fig, ax = canvas(17, 7.4)
    # Inputs
    heads = [("Atoms and topology", "which atoms, how they connect"),
             ("Energy and force model", r"$U(\mathbf{x})$ and $\mathbf{F}=-\nabla U$"),
             ("Ensemble", "conditions held fixed")]
    for k, (head, sub) in enumerate(heads):
        y = 5.15 - 2.35 * k
        box(ax, 0.1, y, 4.9, 2.0, fc=PALE)
        ax.text(0.3, y + 1.78, head, fontsize=13, weight="bold", va="top")
        ax.text(0.3, y + 1.33, sub, fontsize=11, va="top", color="#444444")
        arrow(ax, (5.0, y + 1.0), (6.0, 3.9), style="-", lw=1.6)
    # small molecule
    pts = {"r": (1.2, 5.75), "g1": (1.95, 6.05), "g2": (2.6, 5.55), "b": (3.35, 5.85)}
    for a, b in (("r", "g1"), ("g1", "g2"), ("g2", "b")):
        ax.plot(*zip(pts[a], pts[b]), color=INK, lw=2, zorder=3)
    for key, colour in (("r", RED), ("g1", GREEN), ("g2", GREEN), ("b", BLUE)):
        atom(ax, *pts[key], r=0.2, color=colour)
    # Lennard-Jones sketch
    r = np.linspace(0.94, 2.4, 100)
    v = 4 * (r ** -12 - r ** -6)
    ax.plot(1.0 + 1.3 * (r - 0.94), 3.3 + 0.38 * np.clip(v, -1, 1.3), color=INK, lw=1.6)
    ax.plot([0.95, 3.0], [3.3, 3.3], color=GREY, lw=0.8, ls=":")
    ax.text(3.15, 3.3, r"$V(r)$", fontsize=11, va="center")
    # Fixed-volume container for the ensemble
    ax.add_patch(Rectangle((0.45, 0.55), 1.3, 0.85, fc="white", ec=INK, lw=1.2, ls="--", zorder=2))
    for p in [(0.75, 0.8), (1.1, 1.1), (1.45, 0.75), (0.8, 1.2), (1.45, 1.17)]:
        atom(ax, *p, r=0.1)
    ax.text(2.05, 0.95, "NVE, NVT, NPT, …\n" + r"$N$ atoms, volume $V$," + "\n" + r"energy $E$, temperature $T$," + "\n" + r"or pressure $P$",
            fontsize=10, va="center", linespacing=1.15)
    # Engine
    box(ax, 6.0, 2.75, 3.0, 2.3, lw=2.2)
    ax.text(7.5, 3.45, "MD engine", ha="center", fontsize=15, weight="bold")
    gear(ax, 7.0, 4.3, 0.32)
    gear(ax, 7.85, 4.45, 0.22, teeth=6, color="#c9c9c9")
    ax.text(7.5, 2.45, "workstation or supercomputer", ha="center", va="top", fontsize=10.5, color="#444444")
    ax.add_patch(Ellipse((7.5, 1.35), 2.4, 0.95, fc="none", ec=BLUE, lw=2))
    ax.text(7.5, 1.35, r"$\mathbf{F}=m\mathbf{a}$", ha="center", va="center", fontsize=17, color=BLUE)
    arrow(ax, (9.0, 3.9), (10.1, 3.9), lw=2, ms=18)
    # Results
    box(ax, 10.1, 0.1, 6.8, 7.1, lw=2, r=0.35)
    ax.text(13.5, 6.85, "Results and analysis", ha="center", va="top", fontsize=14, weight="bold")
    mini_axes(ax, 10.55, 3.6, 2.8, 2.65, r"$t$", "energy")
    t = np.linspace(0, 1, 160)
    rng = np.random.default_rng(21)
    wiggle = lambda s: np.convolve(rng.normal(0, s, t.size), np.ones(6) / 6, mode="same")
    event = np.exp(-((t - 0.8) / 0.05) ** 2)
    ax.plot(10.65 + 2.55 * t, 5.35 - 0.45 * event + wiggle(0.18), color=INK, lw=1.4)
    ax.plot(10.65 + 2.55 * t, 4.25 + 0.45 * event + wiggle(0.18), color=ORANGE, lw=1.4)
    ax.text(13.3, 5.35, r"$U$", fontsize=12, va="center")
    ax.text(13.3, 4.3, r"$K$", fontsize=12, va="center", color=ORANGE)
    mini_axes(ax, 13.85, 3.6, 2.75, 2.65, r"$r$", r"$g(r)$")
    rr = np.linspace(0.75, 3.2, 300)
    g = np.where(rr < 0.92, 0, 1 + 2.2 * np.exp(-(rr - 1.0) * 1.9) * np.cos(6.3 * (rr - 1.05)))
    g = np.where((rr > 0.92) & (rr < 1.0), g * (rr - 0.92) / 0.08, g)
    ax.plot(13.85 + 0.95 * (rr - 0.6), 3.6 + 0.62 * g, color=INK, lw=1.4)
    # Atomic event
    ax.text(13.5, 2.55, "atomic events", ha="center", va="bottom", fontsize=12)
    before = np.array([[11.2, 1.5], [11.6, 1.8], [11.6, 1.25], [12.0, 1.55], [11.25, 0.95], [12.0, 1.0]])
    after = np.array([[14.6, 1.55], [15.0, 1.85], [15.0, 1.3], [15.4, 1.6], [14.65, 1.0], [16.3, 0.85]])
    for p in before:
        atom(ax, *p, r=0.17)
    for p in after:
        atom(ax, *p, r=0.17)
    atom(ax, *after[-1], r=0.17, color=RED)
    atom(ax, *before[-1], r=0.17, color=RED)
    arrow(ax, (12.6, 1.4), (14.1, 1.4), lw=1.5, ls="--")
    arrow(ax, (15.75, 0.95), (16.1, 0.88), color=RED, lw=1.3, ms=10)
    save(fig, "L21-md-workflow")


def what_we_know():
    fig, ax = canvas(17, 9.2)
    ax.text(0.1, 8.95, "Atomic coordinates", fontsize=13, weight="bold", va="top")
    ax.text(0.3, 8.15, r"$\mathbf{x}=(x_1,\,y_1,\,z_1,\,x_2,\,\ldots,\,x_N,\,y_N,\,z_N)^{\mathsf{T}}$", fontsize=15, va="center")
    ax.text(0.1, 7.35, "Pairwise energy", fontsize=13, weight="bold", va="top")
    ax.text(0.3, 6.75, r"$V(r_{ij})$", fontsize=17, va="center")
    ax.text(0.3, 6.1, "energy_func(r_ij)", family=MONO, fontsize=11, va="center", color="#444444")
    atom(ax, 3.2, 6.55, r=0.22, color=RED, label="i")
    atom(ax, 5.6, 6.2, r=0.22, color=BLUE, label="j")
    ax.plot([3.2, 5.6], [6.55, 6.2], color=INK, lw=1.4, ls="--", zorder=3)
    ax.text(3.2, 6.2, r"$(x_i,\,y_i,\,z_i)$", fontsize=11, ha="center", va="top")
    ax.text(5.6, 5.85, r"$(x_j,\,y_j,\,z_j)$", fontsize=11, ha="center", va="top")
    ax.text(4.4, 6.48, r"$r_{ij}$", fontsize=13, ha="center", va="bottom")
    ax.text(0.3, 5.0, r"$r_{ij}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2+(z_i-z_j)^2}$", fontsize=14, va="center")
    brace(ax, 8.2, 4.95, 8.95, width=0.45)
    arrow(ax, (8.8, 6.95), (9.6, 6.95), lw=2, ms=18)
    box(ax, 9.8, 4.95, 7.1, 4.0, fc=PALE)
    ax.text(10.05, 8.75, "Total internal energy", fontsize=13, weight="bold", va="top")
    ax.text(10.4, 7.8, r"$U=\sum_{i<j}\,V(r_{ij})$", fontsize=17, va="center")
    ax.text(10.05, 7.0, "Force on each atom, in each direction", fontsize=13, weight="bold", va="top")
    ax.text(10.4, 6.05, r"$F_{x,i}=-\dfrac{\partial U}{\partial x_i},\quad F_{y,i}=-\dfrac{\partial U}{\partial y_i},\quad F_{z,i}=-\dfrac{\partial U}{\partial z_i}$",
            fontsize=15, va="center")
    ax.plot([0.1, 16.9], [4.25, 4.25], color=GREY, lw=1, ls="--")
    # Displacing one coordinate
    ax.text(0.1, 3.95, "Move one coordinate, keep the rest", fontsize=13, weight="bold", va="top")
    for p in [(0.9, 0.7), (1.6, 0.45), (2.4, 0.75), (3.3, 0.5), (4.6, 0.8), (5.3, 1.2)]:
        atom(ax, *p, r=0.15, color="#d6d6d6")
    ax.text(5.8, 0.6, "rest of the atoms", fontsize=11, va="center", color="#555555")
    atom(ax, 1.4, 2.4, r=0.24, fc="white", ls="--")
    atom(ax, 3.4, 2.4, r=0.24, fc="white", ls="--")
    atom(ax, 2.4, 2.4, r=0.26, color=RED, label="i")
    atom(ax, 4.7, 2.0, r=0.24, color=BLUE, label="j")
    arrow(ax, (2.1, 2.4), (1.68, 2.4), lw=1.4, ms=10)
    arrow(ax, (2.7, 2.4), (3.12, 2.4), lw=1.4, ms=10)
    ax.text(1.4, 2.8, r"$x_i-\delta$", ha="center", fontsize=12)
    ax.text(3.4, 2.8, r"$x_i+\delta$", ha="center", fontsize=12)
    ax.text(5.3, 2.9, r"$U(x_i;\ \mathrm{rest})$", fontsize=16, va="center")
    ax.plot([8.6, 8.6], [0.2, 3.8], color=GREY, lw=1, ls="--")
    ax.text(9.0, 3.95, "Finite-difference force", fontsize=13, weight="bold", va="top")
    ax.text(9.2, 3.25, "forward", fontsize=12, va="center", color="#555555")
    ax.text(9.2, 2.55, r"$F_{x,i}\approx-\dfrac{U(x_i+\delta;\,\mathrm{rest})-U(x_i;\,\mathrm{rest})}{\delta}$", fontsize=14, va="center")
    ax.text(9.2, 1.55, "central", fontsize=12, va="center", color="#555555")
    ax.text(9.2, 0.8, r"$F_{x,i}\approx-\dfrac{U(x_i+\delta;\,\mathrm{rest})-U(x_i-\delta;\,\mathrm{rest})}{2\delta}$", fontsize=14, va="center")
    save(fig, "L21-forces-from-energy")


def positions_array():
    fig, ax = canvas(17, 8.6)
    ax.set_ylim(-0.5, 8.6)
    rows, labels = 7, ["atom 0", "atom 1", "⋮", "atom i", "⋮", "atom j", "⋮"]
    marks = {(3, c): "#e8c3bd" for c in range(3)} | {(5, c): "#c1d3ea" for c in range(3)}
    ys, xs = table(ax, 1.4, 7.4, rows, 3, cw=0.65, ch=0.5, marks=marks)
    for x, name in zip(xs, "xyz"):
        ax.text(x, 7.5, name, ha="center", va="bottom", fontsize=13, style="italic")
    hbrace(ax, 1.4, 3.35, 7.95)
    ax.text(2.375, 8.25, "3 columns", ha="center", va="bottom", fontsize=12)
    brace(ax, 1.2, 3.9, 7.4, width=0.3, left=True)
    ax.text(0.75, 5.65, r"$N$" + "\nrows", ha="right", va="center", fontsize=12)
    ax.text(2.375, 3.75, "pos", ha="center", va="top", fontsize=15, family=MONO, weight="bold")
    for y, lab in zip(ys, labels):
        if lab != "⋮":
            arrow(ax, (3.5, y), (4.1, y), lw=1.1, ms=9)
            colour = RED if lab == "atom i" else BLUE if lab == "atom j" else INK
            ax.text(4.2, y, lab, va="center", fontsize=12, color=colour)
        else:
            vdots(ax, 4.5, y)
    brace(ax, 5.3, ys[5] - 0.1, ys[3] + 0.1, width=0.3)
    arrow(ax, (5.75, (ys[3] + ys[5]) / 2), (6.6, (ys[3] + ys[5]) / 2), lw=1.6)
    ax.text(6.8, (ys[3] + ys[5]) / 2 + 0.35, "distance between atoms i and j", fontsize=13, va="bottom", weight="bold")
    ax.text(6.8, (ys[3] + ys[5]) / 2 - 0.15, "r_ij = np.linalg.norm(pos[i] - pos[j])", fontsize=13, va="center", family=MONO)
    # Pair network
    rng = np.random.default_rng(3)
    net = np.array([[7.2, 6.9], [8.1, 7.45], [9.0, 6.95], [8.7, 6.2], [7.6, 6.15], [9.8, 7.4]])
    for a in range(len(net)):
        for b in range(a + 1, len(net)):
            highlight = (a, b) == (1, 3)
            ax.plot(*zip(net[a], net[b]), color=INK if highlight else "#b9b9b9", lw=2.2 if highlight else 0.9, zorder=2)
    for k, p in enumerate(net):
        atom(ax, *p, r=0.16, color=RED if k == 1 else BLUE if k == 3 else GREY)
    ax.text(10.3, 6.95, "every pair\n" + r"$i<j$" + " counted once", fontsize=12, va="center")
    # Energy loop
    ax.plot([0.1, 16.9], [3.0, 3.0], color=GREY, lw=1, ls="--")
    ax.text(0.2, 2.65, "Energy: loop over all pairs", fontsize=13, weight="bold", va="top")
    ax.text(0.5, 1.6, r"$U=V(r_{01})+V(r_{02})+\cdots=\sum_{i<j}V(r_{ij})$", fontsize=15, va="center")
    code = ("def U(pos):\n"
            "    energy = 0.0\n"
            "    for i in range(len(pos)):\n"
            "        for j in range(i + 1, len(pos)):\n"
            "            r_ij = np.linalg.norm(pos[i] - pos[j])\n"
            "            energy += energy_func(r_ij)\n"
            "    return energy")
    box(ax, 8.0, -0.45, 8.9, 3.25, fc=PALE, r=0.15)
    ax.text(8.25, 2.6, code, family=MONO, fontsize=11.5, va="top", linespacing=1.45)
    save(fig, "L21-positions-array")


def numerical_forces():
    fig, ax = canvas(17, 9.2)
    box(ax, 0.1, 0.1, 16.8, 9.0, lw=2.2, r=0.5)
    ax.text(0.45, 8.75, "forces(pos)", family=MONO, fontsize=16, weight="bold", va="top")
    ax.text(3.6, 8.72, "repeat for each of the 3N coordinates", fontsize=13, va="top", color="#444444")
    cw, ch = 0.55, 0.45
    hi = {(1, 1): "#9fbbe0"}
    ys, xs = table(ax, 1.6, 7.3, 5, 3, cw, ch, marks=hi)
    ax.text(2.425, 7.75, "second index: direction", ha="center", fontsize=11, va="bottom")
    for x, name, k in zip(xs, "xyz", range(3)):
        ax.text(x, 7.36, name, ha="center", va="bottom", fontsize=12, style="italic")
        ax.text(x, 4.95, str(k), ha="center", va="top", fontsize=11, family=MONO)
    for y, k in zip(ys, ["0", "1", "2", "3", ""]):
        ax.text(1.45, y, k, ha="right", va="center", fontsize=11, family=MONO)
    vdots(ax, 1.36, ys[-1])
    ax.text(0.5, 6.1, "first\nindex:\natom", ha="center", va="center", fontsize=11)
    ax.text(2.425, 4.4, "pos", ha="center", va="top", fontsize=14, family=MONO, weight="bold")
    ax.text(2.425, 3.9, "pos[1, 1] = y of atom 1", ha="center", va="top", fontsize=11, family=MONO, color=BLUE)
    for sign, y_target, name, colour in (("+", 7.4, "pos_r", RED), ("−", 4.7, "pos_l", BLUE)):
        arrow(ax, (3.4, 6.1), (5.5, y_target - 0.6), lw=1.8, rad=-0.15 if sign == "+" else 0.15, ms=16)
        ax.text(4.0 if sign == "+" else 3.75, 7.15 if sign == "+" else 4.6, sign + r"$\delta$", fontsize=16, ha="center", va="center")
        table(ax, 5.6, y_target, 4, 3, 0.45, 0.35, marks={(1, 1): "#e8c3bd" if sign == "+" else "#c1d3ea"})
        ax.text(6.275, y_target - 1.5, name, ha="center", va="top", family=MONO, fontsize=13, weight="bold")
        arrow(ax, (7.1, y_target - 0.7), (7.9, y_target - 0.7), lw=1.5)
        ax.text(8.0, y_target - 0.7, f"U({name})", family=MONO, fontsize=13, va="center")
    ax.text(5.6, 7.75, "y of atom 1 + δ", fontsize=11, color=RED, va="bottom", family=MONO)
    ax.text(5.6, 5.05, "y of atom 1 − δ", fontsize=11, color=BLUE, va="bottom", family=MONO)
    # Combine and store
    ax.plot([12.4, 12.8, 12.8, 12.4], [6.7, 6.7, 4.0, 4.0], color=INK, lw=1.6)
    arrow(ax, (12.8, 5.35), (12.8, 3.15), lw=1.8, ms=16)
    ax.text(8.75, 2.95, "F[1, 1] = (-(U(pos_r) - U(pos_l))\n"
                       "           / (2 * δ))",
            family=MONO, fontsize=11.5, va="top", ha="left", linespacing=1.4)
    table(ax, 2.0, 3.0, 5, 3, cw, ch, marks={(1, 1): "#9fd1a9"})
    ax.text(1.75, 2.1, "F", family=MONO, fontsize=16, weight="bold", ha="right", va="center")
    arrow(ax, (8.55, 2.55), (3.9, 2.32), lw=1.8, rad=0.0, color=GREEN, ms=16)
    save(fig, "L21-numerical-forces")


def time_step():
    fig, ax = canvas(17, 9.4)
    ax.text(0.1, 9.2, r"At time $t_n$", fontsize=14, weight="bold", va="top")
    ax.text(0.3, 8.35, "pos", family=MONO, fontsize=14, va="center", weight="bold")
    arrow(ax, (1.2, 8.35), (2.0, 8.35), lw=1.5)
    ax.text(2.15, 8.6, "U = internal_energy(pos)", family=MONO, fontsize=13, va="center")
    ax.text(2.15, 8.05, "F = forces(pos)", family=MONO, fontsize=13, va="center")
    ax.text(0.3, 7.35, "vel", family=MONO, fontsize=14, va="center", weight="bold")
    ax.text(1.2, 7.35, "velocities, same N × 3 shape", fontsize=12, va="center", color="#444444")
    cluster = np.array([[10.2, 7.7], [10.85, 8.3], [11.2, 7.55], [10.6, 7.1]])
    vels = np.array([[-0.45, 0.35], [0.1, 0.5], [0.5, -0.1], [-0.15, -0.5]])
    for a in range(4):
        for b in range(a + 1, 4):
            ax.plot(*zip(cluster[a], cluster[b]), color="#c9c9c9", lw=0.9, zorder=2)
    for p, v in zip(cluster, vels):
        atom(ax, *p, r=0.17)
        arrow(ax, p, p + v, lw=1.3, ms=9, color=RED, z=6)
    for x0, name in ((12.4, "pos"), (14.4, "vel")):
        table(ax, x0, 8.7, 4, 3, 0.38, 0.32)
        ax.text(x0 + 0.57, 7.25, name, family=MONO, fontsize=12, ha="center", va="top", weight="bold")
    ax.plot([0.1, 16.9], [6.55, 6.55], color=GREY, lw=1, ls="--")
    for x0, title, needs in ((0.1, "Method 1: forward Euler", "needs the current pos and vel"),
                             (8.6, "Method 2: Størmer–Verlet", "needs the current pos and the previous pos_prev")):
        box(ax, x0, 0.1, 8.3, 6.15, fc=PALE)
        ax.text(x0 + 0.25, 6.0, title, fontsize=14, weight="bold", va="top")
        ax.text(x0 + 0.25, 5.3, needs, fontsize=12, va="center", color="#444444")
    ax.text(0.35, 4.75, "def step_euler(pos, vel, F, dt, m=1.0):\n"
                        "    pos_next = pos + vel * dt\n"
                        "    vel_next = vel + F / m * dt\n"
                        "    return pos_next, vel_next",
            family=MONO, fontsize=11, va="top", linespacing=1.55)
    ax.text(0.35, 2.75, "Both updates use the old values at " + r"$t_n$" + ".", fontsize=12, va="top")
    ax.text(8.85, 4.75, "def step_verlet(pos, pos_prev, F, dt, m=1.0):\n"
                        "    pos_next = 2 * pos - pos_prev + F / m * dt**2\n"
                        "    return pos_next",
            family=MONO, fontsize=11, va="top", linespacing=1.55)
    ax.text(8.85, 3.3, "The velocity is not needed to advance pos.\nRecover it afterwards for the kinetic energy:",
            fontsize=12, va="top", linespacing=1.35)
    ax.text(8.85, 2.3, "vel = (pos_next - pos_prev) / (2 * dt)", family=MONO, fontsize=11, va="top")

    def timeline(x0, x1, ticks, labels, names, colours):
        ax.plot([x0, x1], [0.75, 0.75], color=INK, lw=1.3)
        for x, lab, name, colour in zip(ticks, labels, names, colours):
            ax.plot([x, x], [0.65, 0.85], color=INK, lw=1.3)
            ax.text(x, 0.55, lab, ha="center", va="top", fontsize=12)
            ax.plot(x, 1.15, "o", ms=9, color=colour, mec=INK)
            ax.text(x, 1.4, name, ha="center", va="bottom", family=MONO, fontsize=11)
    timeline(1.0, 7.5, [2.2, 6.2], [r"$t_n$", r"$t_{n+1}$"], ["pos, vel", "pos_next, vel_next"], [GREY, RED])
    arrow(ax, (2.6, 1.15), (5.8, 1.15), lw=1.3, ms=11, rad=0)
    timeline(9.3, 16.4, [10.4, 12.85, 15.3], [r"$t_{n-1}$", r"$t_n$", r"$t_{n+1}$"], ["pos_prev", "pos", "pos_next"], ["#d6d6d6", GREY, RED])
    save(fig, "L21-time-step-methods")


def engine_flow():
    fig, ax = canvas(17, 10.0)
    def node(x, y, w, h, lines, fc="white", fs=13, lw=1.6):
        box(ax, x, y, w, h, fc=fc, lw=lw)
        ax.text(x + w / 2, y + h / 2, lines, ha="center", va="center", family=MONO, fontsize=fs, linespacing=1.5)
    ax.text(0.1, 9.85, "Start", fontsize=14, weight="bold", va="top")
    node(0.2, 7.6, 3.4, 1.7, "pos_guess\nenergy_func", fc=PALE)
    arrow(ax, (3.6, 8.45), (5.6, 8.45), lw=1.8, ms=16)
    ax.text(4.6, 8.6, "minimize", ha="center", va="bottom", fontsize=12)
    ax.text(4.55, 8.3, "jac = -forces", ha="center", va="top", fontsize=9.5, family=MONO, color="#444444")
    node(5.6, 7.95, 2.6, 1.0, "pos_min")
    ax.text(6.9, 7.75, "check: U = −44.3268", ha="center", va="top", fontsize=11, color=GREEN)
    arrow(ax, (8.2, 8.45), (10.6, 8.45), lw=1.8, ms=16)
    ax.text(9.4, 8.6, "set initial\nstate", ha="center", va="bottom", fontsize=12)
    node(10.6, 7.6, 6.2, 1.7, "pos ← pos_min + displacement\nvel ← init_vel", fs=12.5)
    # Run loop
    arrow(ax, (13.7, 7.6), (13.7, 6.9), lw=1.8, ms=16)
    box(ax, 0.2, 2.6, 16.6, 4.3, lw=2.2, r=0.35)
    ax.text(0.5, 6.65, "run(n_steps)", family=MONO, fontsize=15, weight="bold", va="top")
    ax.text(0.6, 5.2, "pos\nvel", family=MONO, fontsize=14, va="center", linespacing=1.6)
    arrow(ax, (1.6, 5.2), (2.6, 5.2), lw=1.6)
    node(2.6, 4.45, 4.8, 1.5, "U = internal_energy(pos)\nF = forces(pos)", fs=12)
    arrow(ax, (7.4, 5.2), (9.0, 5.2), lw=1.6)
    node(9.0, 4.2, 3.6, 2.0, "step_euler\nor\nstep_verlet", fs=12, fc=PALE)
    arrow(ax, (12.6, 5.2), (13.4, 5.2), lw=1.6)
    node(13.4, 4.45, 3.1, 1.5, "record\nt, U, K, pos", fs=12)
    arrow(ax, (14.95, 4.45), (1.0, 4.45), lw=1.6, rad=-0.12, ls="--", color=GREY, ms=14)
    ax.text(8.0, 2.95, "next pos, vel: repeat n_steps times", ha="center", va="center", fontsize=12, color="#555555")
    # End
    arrow(ax, (3.0, 2.6), (3.0, 1.8), lw=1.8, ms=16)
    ax.text(0.1, 1.45, "End", fontsize=14, weight="bold", va="center")
    ax.text(1.3, 1.0, "records over time:", fontsize=13, va="center")
    ax.text(4.6, 1.0, r"$K(t),\quad U(t),\quad E(t)=K(t)+U(t),\quad \mathbf{x}(t),\quad \mathbf{v}(t)$", fontsize=15, va="center")
    save(fig, "L21-md-engine-flow")


VIEW = 4.0  # half-width of the fixed viewing cube, in sigma
VERLET, EULER = "Størmer–Verlet", "forward Euler"


def simulate(check=False):
    app = runpy.run_path(str(ROOT / "activities" / "l21_md_engine_completed.edit.py"))["app"]
    # Batch rendering supplies a submitted configuration explicitly; the browser
    # still waits for the user to click Simulate before doing any calculation.
    _, definitions = app.run(defs={"config": {
        "atoms": 13, "steps": 10, "method": VERLET,
        "dt": 0.005, "added": 0.3,
    }})
    nb = dict(definitions)
    initial_state = nb["initial_state"]

    def run(start, velocity, dt, duration, method):
        if "analyse" in nb:
            step = nb["step_verlet"] if method == VERLET else nb["step_euler"]
            positions = nb["run"](start, velocity, dt, round(duration / dt), step)
        else:
            return nb["run"](start, velocity, dt, duration, method)
        # The revised notebook records positions, then analyses central velocities.
        trajectory = nb["analyse"](positions, dt)
        energy_func = nb["U"]
        trajectory["t"] = np.r_[0.0, trajectory["t"]]
        trajectory["x"] = np.concatenate(([start], trajectory["x"]))
        trajectory["U"] = np.r_[energy_func(start), trajectory["U"]]
        trajectory["K"] = np.r_[0.5 * np.sum(velocity**2), trajectory["K"]]
        return {key: trajectory[key] for key in ("t", "x", "U", "K")}
    np.testing.assert_allclose(nb["U_min"], -44.326801, atol=1e-6)
    start, velocity = initial_state(0.3)
    runs = {
        "verlet": run(start, velocity, 0.005, 10.0, VERLET),
        "euler": run(start, velocity, 0.005, 10.0, EULER),
        "medium": run(start, velocity, 0.04, 10.0, VERLET),
        "large": run(start, velocity, 0.08, 10.0, VERLET),
    }
    steps = [0.005, 0.002, 0.001] if check else [0.005]
    for dt in steps:
        trajectory = runs["verlet"] if dt == 0.005 else run(start, velocity, dt, 10.0, VERLET)
        E = trajectory["U"] + trajectory["K"]
        print(f"Verlet dt={dt}: max |E - E0| = {np.max(np.abs(E - E[0])):.6g}")
    for key in ("euler", "medium", "large"):
        E = runs[key]["U"] + runs[key]["K"]
        radius = np.max(np.linalg.norm(runs[key]["x"][-1], axis=1))
        print(f"{key}: max |E - E0| = {np.max(np.abs(E - E[0])):.6g}, final time {runs[key]['t'][-1]:g}, "
              f"largest final radius {radius:.3g}")
    E = runs["verlet"]["U"] + runs["verlet"]["K"]
    assert np.max(np.abs(E - E[0])) < 0.011 and np.max(runs["euler"]["U"] + runs["euler"]["K"]) - E[0] > 40
    runs["relaxation"] = minimize_cluster(nb)
    return runs


def minimize_cluster(nb):
    """Record accepted BFGS iterates, using the notebook's numerical forces."""
    phi = (1 + np.sqrt(5)) / 2
    shell = []
    for a in [-1, 1]:
        for b in [-phi, phi]:
            shell.extend([[0, a, b], [a, b, 0], [b, 0, a]])
    base = np.vstack([np.zeros((1, 3)), np.array(shell)])
    guess = 0.6 * base + 0.05 * np.random.default_rng(21).normal(size=base.shape)
    guess -= guess.mean(axis=0)
    frames = [guess.copy()]
    energy_func = nb["U"] if "U" in nb else nb["internal_energy"]
    result = nb["optimize"].minimize(
        lambda x: energy_func(x.reshape(-1, 3)),
        guess.ravel(),
        jac=lambda x: -nb["forces"](x.reshape(-1, 3)).ravel(),
        callback=lambda x: frames.append(x.reshape(-1, 3).copy()),
        method="BFGS", options={"gtol": 1e-7},
    )
    energy = np.array([energy_func(p) for p in frames])
    max_force = np.max(np.abs(nb["forces"](frames[-1])))
    np.testing.assert_allclose(energy[-1], -44.326801, atol=1e-6)
    assert max_force < 1e-6 and np.all(np.diff(energy) <= 1e-8)
    print(f"Relaxation: {result.message}; {result.nit} BFGS iterations; "
          f"U {energy[0]:.8f} -> {energy[-1]:.8f}; max |F| {max_force:.3g}")
    return {"x": np.array(frames), "U": energy}


def draw_cluster(ax, p, size, tag=0, view=VIEW):
    ax.set_axis_off()
    inside = np.all(np.abs(p) <= view, axis=1)
    colours = np.array([RED if k == tag else GREY for k in range(len(p))])
    ax.scatter(*p[inside].T, s=size, c=colours[inside], edgecolors=INK, linewidths=0.6, depthshade=True)
    ax.set(xlim=(-view, view), ylim=(-view, view), zlim=(-view, view))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=35)
    return len(p) - np.count_nonzero(inside)


def energy_comparison(runs):
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    for key, title, color in (("euler", "Forward Euler", RED),
                              ("verlet", VERLET, BLUE)):
        trajectory = runs[key]
        ax.plot(trajectory["t"], trajectory["U"] + trajectory["K"],
                color=color, lw=1.8, label=title)
    ax.set(xlabel=r"time $t/\tau$", ylabel=r"total energy $E/\varepsilon$",
           title=r"Same initial state, $\Delta t=0.005$", xlim=(0, 10))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    save(fig, "L21-energy-comparison", svg=False)


def frame_at(trajectory, time):
    return trajectory["x"][np.argmin(np.abs(trajectory["t"] - time))], trajectory["t"][np.argmin(np.abs(trajectory["t"] - time))]


def step_and_energy(runs):
    fig = plt.figure(figsize=(10, 3.8))
    # 0.96 is a stored time for all 3 time steps, before the large-step safety stop.
    snapshot_time = 0.96
    titles = (r"$\Delta t=0.005$", r"$\Delta t=0.04$", r"$\Delta t=0.08$")
    for k, (key, title) in enumerate(zip(("verlet", "medium", "large"), titles)):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        p, time = frame_at(runs[key], snapshot_time)
        np.testing.assert_allclose(time, snapshot_time, atol=1e-12)
        outside = draw_cluster(ax, p, size=230)
        ax.set_title(title, fontsize=14)
        if outside:
            ax.text2D(0.5, 0.02, f"{outside} atoms outside view", transform=ax.transAxes, ha="center", fontsize=11)
        if key in ("verlet", "medium"):
            ax.text2D(0.5, -0.07, "Cluster intact", color=GREEN,
                      transform=ax.transAxes, ha="center", fontsize=13, weight="bold")
        if key == "large" and outside > len(p) // 2:
            ax.text2D(0.5, -0.07, "Cluster disintegrated", color=RED,
                      transform=ax.transAxes, ha="center", fontsize=13, weight="bold")
        print(f"Snapshot {key}: t={time:g}, {outside} atoms outside view")
    fig.suptitle(r"Størmer–Verlet: same initial state, $t=0.96\tau$", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0, right=1, bottom=0.09, top=0.85, wspace=0)
    save(fig, "L21-step-and-energy", svg=False)


def relaxation_movie(runs):
    relaxation = runs["relaxation"]
    frames, energies = relaxation["x"], relaxation["U"]
    fig = plt.figure(figsize=(7, 3.6))
    axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
    fig.subplots_adjust(left=0, right=1, bottom=0.1, top=0.82, wspace=0)
    fig.suptitle("Energy minimization with BFGS", fontsize=14, y=0.98)
    fig.text(0.5, 0.02, "Accepted minimization steps (not physical time)",
             ha="center", fontsize=11)

    def draw(n):
        for ax, p, title, energy in zip(
            axes, (frames[0], frames[n]),
            ("Randomly perturbed guess", f"BFGS iteration {n}"),
            (energies[0], energies[n]),
        ):
            ax.cla()
            draw_cluster(ax, p - p.mean(axis=0), size=160, view=1.65)
            ax.set_title(title, fontsize=12)
            ax.text2D(0.5, -0.02, rf"$U={energy:.5f}\,\varepsilon$",
                      transform=ax.transAxes, ha="center", fontsize=12)

    draw(len(frames) - 1)
    save(fig, "L21-relaxation-final", svg=False, close=False)
    # Repeat the end frames for a readable pause. All displayed structures are actual iterates.
    indices = [0] * 7 + list(range(len(frames))) + [len(frames) - 1] * 12
    FuncAnimation(fig, draw, frames=indices).save(
        ASSETS / "L21-relaxation.gif", writer=PillowWriter(fps=8), dpi=DPI)
    plt.close(fig)
    print("Saved L21-relaxation.gif")


def euler_verlet_movie(runs):
    v, e = runs["verlet"], runs["euler"]
    frames = np.linspace(0, len(v["t"]) - 1, 65).round().astype(int)
    fig = plt.figure(figsize=(6, 2.8))
    axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.86, wspace=0)

    def draw(n):
        for ax, trajectory, title in zip(axes, (v, e), ("Størmer–Verlet", "Forward Euler")):
            ax.cla()
            outside = draw_cluster(ax, trajectory["x"][n], size=60)
            ax.set_title(title, fontsize=10)
            ax.text2D(0.5, 0.02, f"{outside} atoms outside view", transform=ax.transAxes, ha="center", fontsize=8)
        fig.suptitle(f"t = {v['t'][n]:.2f}", fontsize=10, y=0.99)

    draw(frames[-1])
    save(fig, "L21-euler-verlet-final", svg=False, close=False)
    FuncAnimation(fig, draw, frames=frames).save(ASSETS / "L21-euler-verlet.gif", writer=PillowWriter(fps=9), dpi=DPI)
    plt.close(fig)
    print("Saved L21-euler-verlet.gif")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schematics-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="also report the time-step refinement")
    args = parser.parse_args()
    with plt.rc_context(STYLE):
        md_workflow()
        what_we_know()
        positions_array()
        numerical_forces()
        time_step()
        engine_flow()
        if not args.schematics_only:
            runs = simulate(args.check)
            energy_comparison(runs)
            step_and_energy(runs)
            relaxation_movie(runs)
            euler_verlet_movie(runs)


if __name__ == "__main__":
    main()

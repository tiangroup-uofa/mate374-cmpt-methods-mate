"""Draw the L22 figures: Monte Carlo versus molecular dynamics, and each simulation family with its numerical core.

    uv run --locked python scripts/l22_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DPI = 300
INK, GREY = "#231f20", "#5f6670"
RED, BLUE, GREEN, PURPLE = "#c0392b", "#1f5fa8", "#1e8449", "#7d3c98"
STYLE = {"font.size": 15, "font.family": "DejaVu Sans", "svg.fonttype": "none"}


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(ASSETS / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Saved {name}")


def rounded(ax, x, y, w, h, ec, fc="white", alpha=1.0, lw=2.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec="none", alpha=alpha))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="none", ec=ec, lw=lw))


def method_core():
    """Rows from small (bottom) to large (top) length scale: family -> what is modelled -> numerical core."""
    fig = plt.figure(figsize=(13, 7.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 112), ylim=(0, 66))
    ax.axis("off")
    rows = [  # bottom to top
        ("Quantum mechanics\n(DFT)", "electrons around\nfixed nuclei", "Eigenvalue problem\n+ fixed-point iteration", "Å · fs", RED),
        ("Molecular\ndynamics", "Newton's law for\nevery atom", "Ordinary differential\nequations (ODEs)", "nm · ps–ns", BLUE),
        ("Monte Carlo", "random moves of\natoms or chains", "Random sampling\nand averaging", "nm–µm · no time", GREEN),
        ("Continuum\n(flow, diffusion,\nphase field, FEM)", "fields: velocity,\nconcentration, stress", "Partial differential\nequations (PDEs)\n→ sparse Ax = b", "µm–m · s–years", PURPLE),
    ]
    heads = [(22, "Simulation family"), (50, "What is modelled"), (80, "Numerical core"), (104, "Scale")]
    for x, t in heads:
        ax.text(x, 62.5, t, ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    for i, (fam, what, core, scale, c) in enumerate(rows):
        y = 9 + i * 13.5
        rounded(ax, 10, y - 5.4, 24, 10.8, c, fc=c, alpha=0.14)
        ax.text(22, y, fam, ha="center", va="center", fontsize=16, fontweight="bold", color=c, linespacing=1.15)
        ax.text(50, y, what, ha="center", va="center", fontsize=14, color=INK, linespacing=1.2)
        ax.add_patch(FancyArrowPatch((61, y), (66, y), arrowstyle="-|>", mutation_scale=22, color=c, lw=2.4))
        rounded(ax, 66, y - 5.4, 28, 10.8, c, lw=2.4)
        ax.text(80, y, core, ha="center", va="center", fontsize=15, fontweight="bold", color=INK, linespacing=1.2)
        ax.text(104, y, scale, ha="center", va="center", fontsize=13.5, color=GREY)
    # vertical length-scale arrow
    ax.add_patch(FancyArrowPatch((4, 3), (4, 58), arrowstyle="-|>", mutation_scale=30, color=INK, lw=3))
    ax.text(1.2, 30.5, "increasing length and time scale", rotation=90, ha="center", va="center",
            fontsize=15, fontweight="bold", color=INK)
    save(fig, "L22-method-core")


def mc_vs_md():
    """Left: Monte Carlo samples unrelated random chains. Right: MD follows one chain through time."""
    rng = np.random.default_rng(22)
    n = 24

    def chain(steps):
        return np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    # Monte Carlo: independent random configurations
    for k, c in enumerate([GREEN, "#52b06b", "#2c6e3f", "#7cc48f"]):
        ang = rng.uniform(0, 2 * np.pi, n)
        xy = chain(np.c_[np.cos(ang), np.sin(ang)])
        xy -= xy.mean(axis=0)
        xy += np.array([(k % 2) * 12.0, -(k // 2) * 12.0])
        ax1.plot(xy[:, 0], xy[:, 1], "-o", color=c, ms=3.5, lw=1.8, label=f"sample {k + 1}")
    ax1.set_title("Monte Carlo: random samples", fontsize=17, fontweight="bold", color=GREEN)
    ax1.text(0.5, -0.08, "independent configurations, no time\naverage: ⟨A⟩ ≈ (A₁ + A₂ + ⋯ + A_M) / M",
             transform=ax1.transAxes, ha="center", va="top", fontsize=13)
    # Molecular dynamics: one chain, small deterministic moves along a trajectory
    ang = rng.uniform(0, 2 * np.pi, n)
    omega = rng.normal(0, 1, n)
    centre = None
    for k, c in enumerate(["#9ec3ea", "#5d97d6", "#2f6fb8", BLUE]):
        a = ang + omega * 0.25 * k
        xy = chain(np.c_[np.cos(a), np.sin(a)])
        centre = xy.mean(axis=0) if centre is None else centre
        xy = xy - centre + np.array([(k % 2) * 12.0, -(k // 2) * 12.0])
        ax2.plot(xy[:, 0], xy[:, 1], "-o", color=c, ms=3.5, lw=1.8, label=f"t = {k} Δt")
    for p, q in [((5.0, 0.5), (8.0, 0.5)), ((12.0, -4.5), (12.0, -7.5)), ((7.0, -12.0), (4.0, -12.0))]:
        ax2.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, color=INK, lw=2.2))
    ax2.set_title("Molecular dynamics: one trajectory", fontsize=17, fontweight="bold", color=BLUE)
    ax2.text(0.5, -0.08, "the same chain moved by Newton's law\nx(t + Δt) from x(t), v(t), F(t)",
             transform=ax2.transAxes, ha="center", va="top", fontsize=13)
    for ax in (ax1, ax2):
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.75")
    save(fig, "L22-mc-vs-md")


def funnel_scheme():
    """Funnel with pitch depth h above a stem of length l and diameter d."""
    PITCH, GLASS = "#3b2f2a", "#9aa5b1"
    fig, ax = plt.subplots(figsize=(6.2, 6.6), layout="constrained")
    ax.set(xlim=(-6, 7.5), ylim=(-7.5, 8), aspect="equal")
    ax.axis("off")
    # funnel cone (y from 0 to 6) and stem (y from -3 to 0), stem half-width 0.5
    cone_x = [-5, -0.5, -0.5, 0.5, 0.5, 5]
    cone_y = [6, 0, -3, -3, 0, 6]
    ax.fill([-4.2, -0.5, -0.5, 0.5, 0.5, 4.2], [4.9, 0, -3, -3, 0, 4.9], color=PITCH, alpha=0.85, zorder=1)
    ax.plot(cone_x[:3], cone_y[:3], color=GLASS, lw=4, solid_capstyle="round", zorder=2)
    ax.plot(cone_x[3:], cone_y[3:], color=GLASS, lw=4, solid_capstyle="round", zorder=2)
    # pendant drop
    t = np.linspace(0, 2 * np.pi, 100)
    ax.fill(0.75 * np.cos(t), -4.2 + 1.0 * np.sin(t), color=PITCH, alpha=0.85, zorder=1)
    ax.fill([-0.45, 0.45, 0.6, -0.6], [-3, -3, -3.6, -3.6], color=PITCH, alpha=0.85, zorder=1)
    # dimension arrows
    def dim(p, q, label, off, ha="left"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>", mutation_scale=14, color=INK, lw=1.6))
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, fontsize=17, ha=ha, va="center", style="italic")
    dim((5.6, 0), (5.6, 4.9), "h", (0.35, 0))
    dim((1.4, -3), (1.4, 0), "l", (0.35, 0))
    dim((-0.5, -6.2), (0.5, -6.2), "d", (0, -0.6), ha="center")
    ax.plot([0.5, 6.0], [4.9, 4.9], color=GREY, lw=1, ls=":")
    ax.plot([0.5, 6.0], [0, 0], color=GREY, lw=1, ls=":")
    ax.plot([0.5, 1.8], [-3, -3], color=GREY, lw=1, ls=":")
    ax.text(-3.4, 2.4, "pitch\n" + r"$\rho,\ \mu$", color="white", fontsize=15, ha="center", va="center")
    ax.add_patch(FancyArrowPatch((-2.4, 7.6), (-2.4, 5.6), arrowstyle="-|>", mutation_scale=18, color=RED, lw=2.2))
    ax.text(-2.1, 6.9, r"weight $\rho g$", color=RED, fontsize=14, va="center")
    ax.text(-1.0, -4.3, "drop\nvolume V", fontsize=13, ha="right", va="center")
    save(fig, "L22-poiseuille-scheme")


if __name__ == "__main__":
    plt.rcParams.update(STYLE)
    method_core()
    mc_vs_md()
    funnel_scheme()

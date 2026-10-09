"""Check the L15 Newton notebook and draw the static L15 figures.

    uv run --locked python scripts/l15_figures.py [--check-only]
"""
import argparse
from pathlib import Path
import runpy
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DPI = 300
NOTEBOOK = ROOT / "activities" / "l15_newton_relax.edit.py"

# All three panels have compressed bonds and the same two starting positions.
CASES = [(key, 1.0, [(0.3, 0.1), (0.3, 0.4)])
         for key in ("harmonic", "lj", "morse")]
TITLES = {"harmonic": "(a) Harmonic spring", "lj": "(b) Lennard-Jones",
          "morse": "(c) Morse"}


def load():
    """Run the notebook once, then rebuild its model and Newton cells for each bond model."""
    cells = runpy.run_path(str(NOTEBOOK))
    _, base = cells["app"].run()

    def model(key):
        _, d = cells["supplied_functions"].run(
            PAIRS=base["PAIRS"], model_choice=SimpleNamespace(value=key), np=np)
        _, n = cells["newton_loop"].run(
            R0=base["R0"], fixed_atoms=d["fixed_atoms"], gradient=d["gradient"],
            hessian=d["hessian"], np=np)
        def map_data(a):
            _, data = cells["energy_map"].run(a=a, np=np, pair=d["pair"])
            return data

        return SimpleNamespace(**d, newton=n["newton"], map_data=map_data)

    return base, model


def check(model):
    harmonic, lj, morse = model("harmonic"), model("lj"), model("morse")
    # Compressed springs: Newton stops on the saddle at the origin, or reaches a minimum.
    path, _ = harmonic.newton([0.3, 0.1], 1.0)
    assert np.allclose(path[-1], 0, atol=1e-8) and len(path) - 1 == 4
    assert np.allclose(np.linalg.eigvalsh(harmonic.hessian(path[-1], 1.0)), [-0.4, 2])
    path, _ = harmonic.newton([0.3, 0.4], 1.0)
    assert np.allclose(path[-1], [0, -np.sqrt(1.2**2 - 1)], atol=1e-8)
    # Compressed LJ bonds: the same minima, reached only from starts near the y-axis.
    path, _ = lj.newton([0.0, 0.5], 1.0)
    assert np.allclose(path[-1], [0, np.sqrt(1.2**2 - 1)], atol=1e-8)
    path, _ = lj.newton([0.1, 0.6], 1.0)
    assert not np.allclose(path[-1], [0, np.sqrt(1.2**2 - 1)], atol=1e-3)
    # Stretched LJ bonds: an off-centre minimum, or the saddle at the origin.
    path, _ = lj.newton([0.3, 0.1], 1.6)
    assert np.allclose(path[-1], [0.3943, 0], atol=1e-4)
    assert np.all(np.linalg.eigvalsh(lj.hessian(path[-1], 1.6)) > 0)
    path, _ = lj.newton([0.15, 0.0], 1.6)
    assert np.allclose(path[-1], 0, atol=1e-8)
    # Morse: pure Newton steps leave the region.
    path, message = morse.newton([0.3, 0.3], 1.25)
    assert message.startswith("Stopped: a Newton step sent")
    print("L15 Newton checks passed.")


@matplotlib.rc_context({"font.size": 14, "font.family": "Arial", "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold"})
def energy_sections(harmonic):
    energy, hessian = harmonic.energy, harmonic.hessian
    a = 1.0
    y_min = np.sqrt(1.2**2 - a**2)
    points = np.linspace(-0.85, 0.85, 181)
    X, Y = np.meshgrid(points, points)
    Z = np.array([[energy(np.array([x, y]), a) for x in points] for y in points])
    fig, (ax, cut) = plt.subplots(1, 2, figsize=(9, 4.3), layout="constrained")
    ax.contour(X, Y, Z, levels=[0.002, 0.01, 0.025, 0.04, 0.08, 0.16, 0.32], colors="#8a8f98")
    ax.axhline(0, color="#231f20", ls="--")
    ax.axvline(0, color="#b5473a", ls="--")
    ax.plot([0, 0], [-y_min, y_min], "o", color="#b5473a")
    ax.plot(0, 0, "x", color="#231f20", ms=9, mew=2)
    label = {"fontsize": 13, "bbox": {"facecolor": "white", "edgecolor": "none", "pad": 1.5}}
    for y in (y_min, -y_min):
        ax.text(0.07, y, "true minimum", color="#b5473a", va="center", **label)
    ax.text(0.07, 0.07, "saddle point", color="#231f20", va="bottom", **label)
    ax.set(xlabel="x / a", ylabel="y / a", aspect="equal", title="Energy contours")

    s = np.linspace(-1.15, 1.15, 231)
    cut.plot(s, [energy(np.array([v, 0]), a) for v in s], color="#231f20")
    cut.plot(s, [energy(np.array([0, v]), a) for v in s], color="#b5473a")
    cut.text(0.71, 0.97, "along x\n(y = 0)", transform=cut.transAxes,
             color="#231f20", fontsize=13, va="top")
    cut.text(-1.0, 0.068, "along y (x = 0)", color="#b5473a", fontsize=13)
    E0 = energy(np.zeros(2), a)
    cut.plot(0, E0, "x", color="#231f20", ms=9, mew=2)
    cut.plot([-y_min, y_min], [0, 0], "o", color="#b5473a", clip_on=False, zorder=5)
    cut.annotate("saddle point", (0.01, E0 + 0.002), xytext=(0.75, 0.1),
                 ha="center", va="center", fontsize=13,
                 arrowprops={"arrowstyle": "->", "color": "#231f20", "shrinkB": 4})
    # Hessian at the saddle: a 2x2 array laid out in points from an anchor in axes
    # coordinates. Diagonal entries match the colours of the two cuts.
    H = hessian(np.zeros(2), a)
    points_from = (matplotlib.transforms.Affine2D().scale(1/72) + fig.dpi_scale_trans
                   + matplotlib.transforms.ScaledTranslation(0.04, 0.68, cut.transAxes))
    cut.text(0, 34, r"$\mathbf{H}(0,0)=$", transform=points_from, fontsize=14, va="center")
    colours = [["#231f20", "#231f20"], ["#231f20", "#b5473a"]]
    for i in range(2):
        for j in range(2):
            cut.text(26 + 36*j, 9 - 18*i, f"{H[i, j]:g}".replace("-", "−"),
                     transform=points_from, color=colours[i][j],
                     ha="right", va="center", fontsize=14)
    for xb, side in ((0, 1), (68, -1)):
        cut.plot([xb + 4*side, xb, xb, xb + 4*side], [19, 19, -19, -19],
                 transform=points_from, color="#231f20", lw=1.2, clip_on=False)
    cut.set(xlabel="Displacement from origin / a", ylabel="Energy / (k a²)",
            xlim=(-1.15, 1.15), ylim=(0, 0.22), title="Energy cuts through the saddle")
    return fig


@matplotlib.rc_context({"font.size": 15, "font.family": "Arial"})
def newton_paths(model):
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.86, bottom=0.25, wspace=0.23)
    colours = ["#b5473a", "#1f77b4"]
    for ax, (key, a, starts) in zip(axes, CASES):
        m = model(key)
        data = m.map_data(a)
        ax.contourf(data["X"], data["Y"], data["Z"],
                    levels=data["levels"], cmap="viridis", extend="max")
        ax.contour(data["X"], data["Y"], data["Z"],
                   levels=data["levels"][2:-1:3], colors="white",
                   linewidths=0.4, linestyles="solid")
        for colour, start in zip(colours, starts):
            path, message = m.newton(start, a)
            assert message.startswith("Converged:"), (key, start, message)
            ax.plot(path[:, 0], path[:, 1], "-", color=colour, lw=1.8, zorder=3)
            ax.plot(*start, "D", color=colour, mec="white", mew=0.6, ms=7, zorder=4)
            ax.plot(*path[-1], "*", color=colour, mec="#231f20",
                    mew=0.5, ms=13, zorder=5)
        ax.plot([-a, a], [0, 0], "s", color="#231f20", ms=7, zorder=5)
        ax.set(xlim=(-1.9, 1.9), ylim=(-1.4, 1.4), aspect="equal",
               xlabel="x", ylabel="y", title=TITLES[key])
        ax.tick_params(labelsize=12)
    handles = [
        Line2D([], [], color=colours[0], lw=2, label="Start (0.3, 0.1)"),
        Line2D([], [], color=colours[1], lw=2, label="Start (0.3, 0.4)"),
        Line2D([], [], ls="", marker="D", color="#5e5e5e", ms=7, label="Start"),
        Line2D([], [], ls="", marker="*", color="#5e5e5e", ms=12, label="Stop"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, fontsize=14, bbox_to_anchor=(0.53, 0.005))
    return fig

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    _, model = load()
    check(model)
    if not args.check_only:
        figures = [(energy_sections(model("harmonic")), "L15-energy-sections.png"),
                   (newton_paths(model), "L15-newton-paths.png")]
        for fig, name in figures:
            fig.savefig(ASSETS / name, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved assets/{name}")
    plt.close("all")


if __name__ == "__main__":
    main()

"""Validate only the two L14 notebooks and regenerate their static figures.

    uv run --locked python scripts/l14_figures.py [--check-only]
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DPI = 300


def load(name):
    app = runpy.run_path(str(ROOT / "activities" / f"{name}.edit.py"))["app"]
    _, definitions = app.run()
    return dict(definitions)


def svg_mm_canvas(path):
    """Keep Matplotlib's editable text, with an Inkscape-friendly millimetre canvas."""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(path)
    root = tree.getroot()
    _, _, width, height = map(float, root.attrib["viewBox"].split())
    scale = 25.4/72
    root.set("width", f"{width*scale:.6f}mm")
    root.set("height", f"{height*scale:.6f}mm")
    root.set("viewBox", f"0 0 {width*scale:.6f} {height*scale:.6f}")
    group = ET.Element("{http://www.w3.org/2000/svg}g", {"transform": f"scale({scale})"})
    for child in list(root):
        root.remove(child)
        group.append(child)
    root.append(group)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def numerical_jacobian(fun, x, h=1e-6):
    return np.column_stack([(fun(x+h*e)-fun(x-h*e))/(2*h) for e in np.eye(len(x))])


def sweep(d, n, stop=2.6):
    positions, evaluate, valid = d["chain_model"](n)
    guess = d["r0"]*np.delete(np.arange(n)-n//2, n//2)

    def solve(x, f):
        return d["newton_system"](lambda y: evaluate(y, f)[1],
                                  lambda y: evaluate(y, f)[2], x, valid=valid)

    x, _, ok, _ = solve(guess, 0)
    assert ok
    rest = x.copy()
    loads, states = [0.0], [x.copy()]
    for step in (0.02, 0.002, 0.0002, 0.00002):
        while loads[-1] + step <= stop:
            f = loads[-1] + step
            candidate, _, ok, _ = solve(x, f)
            if not ok:
                break
            try:
                np.linalg.cholesky(-evaluate(candidate, f)[2])
            except np.linalg.LinAlgError:
                break
            x = candidate
            loads.append(f)
            states.append(x.copy())
    return positions, evaluate, rest, np.array(loads), np.array(states)


def check(simple, d):
    print("Test-function roots:", simple["reference_roots"])
    assert simple["converged"]
    # Check the displayed L13-style pivot/elimination calculation.
    x0, f0, j0, delta0 = simple["history"][0]
    np.testing.assert_allclose(x0, [2, 2])
    augmented = np.column_stack((j0, -f0))[[1, 0]].copy()
    augmented[1] -= augmented[1, 0]/augmented[0, 0]*augmented[0]
    np.testing.assert_allclose(augmented[1], [0, 2.632224, 0.995760], atol=5e-7)
    np.testing.assert_allclose(delta0, [1.421400, 0.378296], atol=5e-7)
    for x in simple["reference_roots"]:
        np.testing.assert_allclose(simple["residual"](x), 0, atol=1e-8)
        np.testing.assert_allclose(simple["jacobian"](x), numerical_jacobian(simple["residual"], x), atol=1e-6)
    _, _, ok, message = simple["newton_system"](simple["residual"], simple["jacobian"], [0, 2])
    assert not ok and "Singular" in message
    assert d["converged"] and d["library_accepted"] and d["library_residual"] < 1e-9
    assert d["ok7"] and d["seven_residual"] < 1e-9 and d["reached7"] == 2.0
    np.testing.assert_allclose(d["seven_strains"], d["seven_strains"][::-1], atol=1e-9)
    for f in (0, 1, 2, 2.43, 2.436):
        x, history, ok, _, _, _ = d["load_trimer"]([-1.12, 1.12], f)
        assert ok, f
        np.testing.assert_allclose(d["force"](x, f), 0, atol=1e-8)
        np.testing.assert_allclose(d["force_jacobian"](x), numerical_jacobian(lambda y: d["force"](y, f), x), atol=1e-6)
        np.testing.assert_allclose(x[0], -x[1], atol=1e-9)
        print(f"Trimer f={f}: x={x}, J={d['force_jacobian'](x).tolist()}, cond={np.linalg.cond(d['force_jacobian'](x)):.6g}")
    # Check the energy/force sign away from symmetric coordinates too.
    x = np.array([-1.17, 1.21])
    gradient = numerical_jacobian(lambda y: np.array([d["energy"](y)]), x)[0]
    np.testing.assert_allclose(d["force"](x, 0), -gradient, atol=1e-6)
    np.testing.assert_allclose(d["force_jacobian"](x), numerical_jacobian(lambda y: d["force"](y, 2), x), atol=1e-6)
    for use_continuation in (False, True):
        for f in (0, 2):
            assert d["load_trimer"]([-1.12, 1.12], f, use_continuation)[2]
        assert not d["load_trimer"]([-1.12, 1.12], 2.6, use_continuation)[2]
    # The handwritten generic Newton loop solves a linear force law in one correction.
    K = np.array([[2., -1.], [-1., 2.]])
    _, history, ok, _ = d["newton_system"](lambda x: np.array([1., 2.])-K@x, lambda x: -K, [0., 0.])
    assert ok and len(history)-1 == 1
    pos3, eval3, _ = d["chain_model"](3)
    np.testing.assert_allclose(eval3(x, 2)[1], d["force"](x, 2))
    np.testing.assert_allclose(eval3(x, 2)[2], d["force_jacobian"](x))
    trimer_s = brentq(lambda r: d["d2V"](r)+2*d["d2V"](2*r), 1.2, 1.3)
    trimer_max = d["dV"](trimer_s)+d["dV"](2*trimer_s)
    print(f"Trimer limit r={trimer_s:.9f}, f={trimer_max:.9f}, J={d['force_jacobian']([-trimer_s,trimer_s])}")
    print("Default trimer:", d["chain_x"], "energy:", d["energy"](d["chain_x"]))
    return trimer_s, trimer_max


@matplotlib.rc_context({"font.size": 12, "font.family": "Arial"})
def force_models(d):
    r0 = d["r0"]
    k0 = d["d2V"](r0)
    r = np.linspace(1.0, 1.8, 600)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), layout="constrained")
    axes[0].plot(r, d["V"](r), color="#b5473a", lw=2, label="Lennard–Jones")
    axes[0].plot(r, -1+0.5*k0*(r-r0)**2, "--", color="#231f20", lw=2, label="Harmonic")
    axes[0].set(xlabel="Separation r (σ)", ylabel="Pair energy V (ε)", ylim=(-1.1, 0.7))
    axes[1].plot(r-r0, d["dV"](r), color="#b5473a", lw=2, label="Lennard–Jones")
    axes[1].plot(r-r0, k0*(r-r0), "--", color="#231f20", lw=2, label="Harmonic")
    axes[1].set(xlabel=r"Extension $r-r_0$ (σ)", ylabel="Restoring force (ε/σ)", ylim=(-1, 4), xlim=(-0.02, 0.65))
    peak_extension = d["r_s"] - r0
    peak_force = d["dV"](d["r_s"])
    axes[1].hlines(peak_force, -0.02, 0.65, colors="#5e5e5e", linestyles=":", lw=1)
    axes[1].vlines(peak_extension, -1, peak_force, colors="#5e5e5e", linestyles=":", lw=1)
    axes[1].plot(peak_extension, peak_force, "o", color="#b5473a", ms=5)
    axes[1].text(0.30, peak_force+0.13,
                 rf"$f_{{\max}}={peak_force:.3f}$ ε/σ", fontsize=10)
    axes[1].text(peak_extension+0.02, -0.65,
                 rf"$r_s-r_0={peak_extension:.3f}$ σ", fontsize=10)
    for label, ax in zip("ab", axes):
        ax.text(-0.16, 1.02, label, transform=ax.transAxes, weight="bold", fontsize=18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=10)
    return fig


@matplotlib.rc_context({"font.size": 12, "font.family": "Arial", "svg.fonttype": "none"})
def chain_pull(d, result):
    positions, evaluate, rest, loads, states = result
    extensions = (states[:, -1]-states[:, 0])-(rest[-1]-rest[0])
    b = np.zeros(len(rest))
    b[0], b[-1] = -1, 1
    linear = np.linalg.solve(-evaluate(rest, 0)[2], b)
    compliance = linear[-1]-linear[0]
    fig, ax = plt.subplots(figsize=(7.1, 5.4), layout="constrained")
    ax.plot(extensions, loads, color="#b5473a", lw=2, label="LJ chain, N = 7")
    ax.plot(loads*compliance, loads, "--", color="#231f20", lw=2, label="Harmonic tangent at zero load")
    ax.plot(extensions[-1], loads[-1], "x", color="#b5473a", mew=2, ms=9)
    ax.set(xlabel="Total chain extension (σ)", ylabel="Outward pull f (ε/σ)",
           ylim=(0, 3.7), xlim=(0, 1.2*extensions[-1]))
    ax.legend(loc="lower right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    inset = ax.inset_axes([0.04, 0.73, 0.92, 0.24])
    p = positions(states[-1])
    strain = np.diff(p)/np.diff(positions(rest))-1
    inset.plot(p, np.zeros(7), color="#8a8f98", lw=2)
    inset.scatter(p, np.zeros(7), s=130, c=["#b5473a"]*3+["#8a8f98"]+["#b5473a"]*3, edgecolors="#231f20", zorder=3)
    for i, value in enumerate(strain):
        inset.text((p[i]+p[i+1])/2, -0.25, f"{100*value:.1f}%", ha="center", fontsize=10)
    for i, direction in [(0, -1), (-1, 1)]:
        inset.annotate("", xy=(p[i]+direction*0.8, 0), xytext=(p[i]+direction*0.2, 0),
                       arrowprops={"arrowstyle": "->", "color": "#b5473a", "lw": 1.5})
    inset.text(0, 0.4, f"Last accepted load: {loads[-1]:.5f} ε/σ", ha="center", fontsize=11)
    inset.set(xlim=(p[0]-1, p[-1]+1), ylim=(-0.6, 0.75))
    inset.axis("off")
    print("Seven-atom limit:", loads[-1], "bond strains:", strain)
    return fig


@matplotlib.rc_context({"font.size": 12, "font.family": "Arial"})
def local_stiffness(d):
    r = np.linspace(d["r0"], 1.7, 400)
    fig, ax = plt.subplots(figsize=(6.5, 3.2), layout="constrained")
    ax.plot(r, d["d2V"](r), color="#b5473a", lw=2)
    ax.axhline(0, color="#8a8f98", lw=1)
    ax.axvline(d["r_s"], color="#8a8f98", ls=":")
    ax.text(d["r_s"]+0.015, 32, r"$r_s$: maximum restoring force", fontsize=11)
    ax.set(xlabel="Pair separation r (σ)", ylabel="Local stiffness V″(r) (ε/σ²)")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


@matplotlib.rc_context({"font.size": 12, "font.family": "Arial"})
def newton_linearization(simple):
    """Plot the exact zero curves and the affine equations actually solved."""
    u, v = np.meshgrid(np.linspace(1.75, 3.8, 300), np.linspace(1.7, 3.35, 260))
    exact = [v-np.cosh(u/2), 9*u**2+25*v**2-225]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.0), layout="constrained")
    for m, ax in enumerate(axes):
        x, f, J, delta = simple["history"][m]
        next_x = x+delta
        for i, color in enumerate(("#1f77b4", "#b5473a")):
            ax.contour(u, v, exact[i], levels=[0], colors=[color], linewidths=1.8)
            affine = f[i]+J[i, 0]*(u-x[0])+J[i, 1]*(v-x[1])
            ax.contour(u, v, affine, levels=[0], colors=[color],
                       linewidths=1.4, linestyles="--")
        np.testing.assert_allclose(f+J@delta, 0, atol=1e-12)
        ax.scatter(*simple["reference_roots"][1], marker="*", s=100, color="#231f20", zorder=5)
        ax.plot(*x, "o", color="#231f20", ms=5)
        ax.plot(*next_x, "o", mec="#231f20", mfc="white", ms=6, zorder=6)
        ax.annotate("", xy=next_x, xytext=x,
                    arrowprops={"arrowstyle": "->", "color": "#5e5e5e", "lw": 1.3,
                                "shrinkA": 5, "shrinkB": 5})
        # Offset labels to keep them clear of both curves and the correction arrow.
        for point, label, offset in [
            (x, rf"$\mathbf{{x}}^{{({m})}}$", (0, -24)),
            (next_x, rf"$\mathbf{{x}}^{{({m+1})}}$", (2, 15)),
        ]:
            ax.annotate(label, point, xytext=offset, textcoords="offset points", fontsize=11)
        ax.set(xlim=(1.75, 3.8), ylim=(1.7, 3.35), xlabel="$x_1$", ylabel="$x_2$")
        ax.set_title(f"Iteration {m}", fontsize=12)
        ax.text(-0.19, 1.04, "ab"[m], transform=ax.transAxes, weight="bold", fontsize=18)
        ax.spines[["top", "right"]].set_visible(False)
    # The same colours identify the same equations in the surface figure.
    axes[0].plot([], [], color="#1f77b4", label="$F_1 = 0$")
    axes[0].plot([], [], color="#b5473a", label="$F_2 = 0$")
    axes[0].legend(loc="upper left", fontsize=10, frameon=False)
    return fig


@matplotlib.rc_context({"font.size": 12, "font.family": "Arial"})
def trimer_limit(d, rlim, fmax):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8), layout="constrained")
    r = np.linspace(1.19, 1.31, 500)
    force = d["dV"](r)+d["dV"](2*r)
    axes[0].plot(r, force, color="#b5473a", lw=2)
    axes[0].axhline(2.45, color="#231f20", ls="--", lw=1.2)
    axes[0].text(1.195, 2.459, "Applied pull: 2.45", fontsize=10)
    axes[0].plot(rlim, fmax, "o", color="#b5473a", ms=5)
    axes[0].annotate(f"Maximum: {fmax:.6f}", (rlim, fmax),
                     xytext=(1.205, 2.32), fontsize=10,
                     arrowprops={"arrowstyle": "-", "color": "#5e5e5e"})
    axes[0].set(xlabel="Bond separation r (σ)", ylabel="Outward pull f (ε/σ)",
                xlim=(1.19, 1.31), ylim=(2.25, 2.48))
    gap = np.geomspace(1e-1, 1e-6, 400)
    cond = [np.linalg.cond(d["force_jacobian"]([-s, s])) for s in rlim-gap]
    axes[1].loglog(gap, cond, color="#b5473a", lw=2)
    axes[1].set(xlabel="Gap to limiting separation (σ)", ylabel=r"Condition number $\kappa_2(\mathbf{J})$",
                xlim=(1e-1, 1e-6))
    axes[1].set_xticks([1e-1, 1e-3, 1e-6])
    axes[1].text(0.04, 0.88, "Approaching the limit →", transform=axes[1].transAxes, fontsize=10)
    for label, ax in zip("ab", axes):
        ax.text(-0.22, 1.04, label, transform=ax.transAxes, weight="bold", fontsize=18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    simple = load("l14_newton_system")
    d = load("l14_bond_chain")
    trimer_s, trimer_max = check(simple, d)
    results = {n: sweep(d, n) for n in (3, 5, 7, 9)}
    assert abs(results[3][3][-1]-trimer_max) < 3e-5
    assert abs(results[7][3][-1]-2.443) < 1e-3
    positions, evaluate, rest, _, states = results[7]
    np.testing.assert_allclose(evaluate(states[-1], results[7][3][-1])[2],
                               numerical_jacobian(lambda x: evaluate(x, 2)[1], states[-1]), atol=1e-6)
    np.testing.assert_allclose(positions(states[-1]), -positions(states[-1])[::-1], atol=1e-8)
    if not args.check_only:
        figures = {
            "L14-newton-system": simple["system_figure"],
            "L14-newton-linearization": newton_linearization(simple),
            "L14-trimer-limit": trimer_limit(d, trimer_s, trimer_max),
            "L14-trimer-newton": d["chain_figure"],
            "L14-force-models": force_models(d),
            "L14-chain-pull": chain_pull(d, results[7]),
            "L14-local-stiffness": local_stiffness(d),
        }
        for name, fig in figures.items():
            fig.savefig(ASSETS / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
            if name == "L14-chain-pull":
                with matplotlib.rc_context({"svg.fonttype": "none"}):
                    fig.savefig(ASSETS / f"{name}.svg", dpi=DPI, bbox_inches="tight", facecolor="white")
                svg_mm_canvas(ASSETS / f"{name}.svg")
            print(f"Saved assets/{name}.png")
    plt.close("all")
    print("L14 checks passed.")


if __name__ == "__main__":
    main()

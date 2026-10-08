"""Check the L16 Hessian and cluster notebooks and draw the static L16 figures.

The LJ38 multistart and free-energy figures are kept for the LJ38 project draft.

    uv run --locked python scripts/l16_figures.py [--check-only]
"""
import os

# Tiny L-BFGS-B vectors run far slower with threaded OpenBLAS; the browser is single-threaded.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DPI = 300
ICOSAHEDRAL_LJ38 = -173.252378  # lowest icosahedral-funnel minimum, Doye, Miller and Wales (1999)


def load(name="l16_lj_cluster"):
    app = runpy.run_path(str(ROOT / "activities" / f"{name}.edit.py"))["app"]
    _, definitions = app.run()
    return dict(definitions)


def check(spring, cluster):
    # theta = 45°, l0 = 1.2a: H = k [[0.8, 1.2], [1.2, 0.8]], eigenvalues -0.4k and 2k.
    np.testing.assert_allclose(spring["H0"], [[0.8, 1.2], [1.2, 0.8]], atol=1e-6)
    np.testing.assert_allclose(spring["values"], [-0.4, 2.0], atol=1e-6)
    assert not spring["is_positive_definite"](spring["H0"])
    assert spring["is_positive_definite"](spring["H_min"])
    # A1 square: two negative modes; following mode 1 passes the rhombus saddle to the tetrahedron.
    assert cluster["square_counts"] == {"negative": 2, "zero": 6, "positive": 4}
    energies = [cluster["lj_energy"](X) for X in cluster["follow_path"]]
    np.testing.assert_allclose(energies, [-4.480620, -5.073421, -6.0], atol=1e-6)
    print("Square eigenvalues:", np.round(cluster["square_values"], 3))


def spring_figure(spring):
    energy, fixed_atoms, hessian = spring["energy"], spring["fixed_atoms"], spring["hessian"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    xs = np.linspace(-1.5, 1.5, 241)
    Xg, Yg = np.meshgrid(xs, xs)
    for ax, theta in zip(axes, (0, 45)):
        fixed = fixed_atoms(theta)
        Z = np.array([[energy(np.array([x, y]), fixed, 1.2) for x in xs] for y in xs])
        ax.contourf(Xg, Yg, np.log10(Z + 1e-3), levels=30, cmap="Greys_r", alpha=0.6)
        ax.contour(Xg, Yg, Z, levels=np.linspace(0, 1.0, 11), colors="white", linewidths=0.5)
        ax.plot(*fixed.T, "s", color="black", ms=9)
        H = hessian(np.zeros(2), fixed, 1.2)
        values, vectors = np.linalg.eigh(H)
        for lam, v, colour in zip(values, vectors.T, ["tab:red", "tab:blue"]):
            ax.annotate("", xy=0.6*v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=colour, lw=2.5))
            ax.plot([], [], color=colour, lw=2.5, label=f"λ = {lam:+.2f}k")
        title = (f"θ = {theta}°:  H = k[[{H[0, 0]:.1f}, {H[0, 1]:.1f}], [{H[1, 0]:.1f}, {H[1, 1]:.1f}]]")
        ax.set(xlabel="x / a", ylabel="y / a", aspect="equal", title=title)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def escape(d, X, max_escapes=5):
    """Repeat the notebook's downhill-mode step until no negative eigenvalue remains."""
    for _ in range(max_escapes):
        values, vectors = d["normal_modes"](X)
        if d["classify"](values)[0] == 0:
            return X
        X, _ = d["relax"](X + 0.1*vectors[:, 0].reshape(-1, 3))
    return X


def check_cluster(d):
    relax, modes, classify, energy = d["relax"], d["normal_modes"], d["classify"], d["lj_energy"]
    ref = d["REFERENCE_ENERGY"]
    assert d["gradient_ok"]

    # Default notebook state: N = 38 from the octahedral hole is the truncated octahedron.
    assert d["N"] == 38
    assert abs(d["E_min"] - ref[38]) < 1e-5
    assert classify(d["eigenvalues"])[3] == "a local minimum"

    # Atom-centred N = 13 and N = 55 fcc fragments relax to cuboctahedral saddles.
    results = {}
    for N in (13, 55):
        X, _ = relax(d["fcc_cluster"](N, centre="atom"))
        values, _ = modes(X)
        assert classify(values)[0] >= 1
        X_final = escape(d, X)
        assert abs(energy(X_final) - ref[N]) < 1e-5
        results[N] = (X, values, X_final, modes(X_final)[0])

    # Step 6: harmonic free energies of the two LJ38 funnels.
    assert abs(d["delta_E"] - 0.676048) < 1e-5
    assert abs(d["delta_S"] - 2.1385) < 1e-3
    assert abs(d["T_cross"] - 0.3161) < 1e-3
    print(f"LJ38 ico - TO: dE = {d['delta_E']:.4f} eps, dS = {d['delta_S']:.3f} kB, "
          f"T* = {d['T_cross']:.3f} eps/kB")

    vib = d["CM1_PER_UNIT"]*np.sqrt(d["eigenvalues"][6:])
    print(f"LJ38 truncated octahedron: E = {d['E_min']:.6f} eps, "
          f"wavenumbers {vib.min():.1f}-{vib.max():.1f} cm^-1")
    for N, (X, values, _, final_values) in results.items():
        print(f"LJ{N} cuboctahedron: E = {energy(X):.6f} eps, lowest eigenvalue {values[0]:.3f}; "
              f"after escape {ref[N]:.6f} eps, lowest vibrational eigenvalue {final_values[6]:.3f}")
    return results


def draw_cluster(ax, X, title):
    X = X - X.mean(axis=0)
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    for i, j in zip(*np.triu_indices(len(X), k=1)):
        if d[i, j] < 1.25*2**(1/6):
            ax.plot(*X[[i, j]].T, color="#999999", lw=0.8, zorder=1)
    ax.scatter(*X.T, s=170, c=X[:, 0] + X[:, 1], cmap="Blues", edgecolors="#1f3b57",
               linewidths=0.6, depthshade=True, zorder=2)
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    lim = np.abs(X).max()*1.05
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.view_init(elev=18, azim=35)
    ax.set_axis_off()


def multistart_energies(d, N, n_starts):
    runs = [d["relax"](d["random_cluster"](N, seed=s)) for s in range(n_starts)]
    return np.array([r.fun for _, r in runs]), [X for X, _ in runs]


def figures(d, results, n_starts=60):
    E13, _ = multistart_energies(d, 13, n_starts)
    E38, X38 = multistart_energies(d, 38, n_starts)
    best = int(np.argmin(E38))
    print(f"{n_starts} random starts: LJ13 hits {np.sum(E13 < d['REFERENCE_ENERGY'][13] + 1e-4)}; "
          f"LJ38 lowest {E38.min():.6f}, median {np.median(E38):.3f}")

    fig = plt.figure(figsize=(8, 4))
    draw_cluster(fig.add_subplot(1, 2, 1, projection="3d"), d["X_min"],
                 f"Truncated octahedron from the fcc start\nE = {d['E_min']:.3f} ε")
    draw_cluster(fig.add_subplot(1, 2, 2, projection="3d"), X38[best],
                 f"Best of {n_starts} random starts\nE = {E38[best]:.3f} ε")
    fig.tight_layout()
    fig.savefig(ASSETS / "L16-lj38-structures.png", dpi=DPI, bbox_inches="tight", facecolor="white")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, E, N in [(axes[0], E13, 13), (axes[1], E38, 38)]:
        ax.hist(E, bins=20, color="tab:blue", alpha=0.8, label="Random starts")
        ax.axvline(d["REFERENCE_ENERGY"][N], color="black", ls="--", label="Lowest known minimum")
        if N == 38:
            ax.axvline(ICOSAHEDRAL_LJ38, color="tab:orange", ls=":", lw=2,
                       label="Lowest icosahedral minimum")
        ax.set(xlabel="Final energy (ε)", ylabel="Number of starts",
               title=f"LJ$_{{{N}}}$: {n_starts} relaxations from random positions")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "L16-multistart.png", dpi=DPI, bbox_inches="tight", facecolor="white")

    X_cubo, cubo_values, X_ico, ico_values = results[13]
    fig = plt.figure(figsize=(10, 3.6))
    draw_cluster(fig.add_subplot(1, 3, 1, projection="3d"), X_cubo,
                 f"Cuboctahedron, E = {d['lj_energy'](X_cubo):.3f} ε")
    draw_cluster(fig.add_subplot(1, 3, 2, projection="3d"), X_ico,
                 f"Icosahedron, E = {d['lj_energy'](X_ico):.3f} ε")
    ax = fig.add_subplot(1, 3, 3)
    k = np.arange(len(cubo_values))
    ax.plot(k, cubo_values, "o", ms=4, color="tab:red", label="Cuboctahedron")
    ax.plot(k, ico_values, "s", ms=4, mfc="none", color="tab:blue", label="Icosahedron")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Mode index k (ascending)", ylabel="Hessian eigenvalue λ (ε/σ²)",
           title="LJ$_{13}$ Hessian eigenvalues, 3N = 39")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "L16-lj13-hessian.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    d["free_energy_figure"].savefig(ASSETS / "L16-free-energy.png", dpi=DPI, bbox_inches="tight",
                                    facecolor="white")
    for name in ("L16-free-energy.png", "L16-lj38-structures.png", "L16-multistart.png", "L16-lj13-hessian.png"):
        print(f"Saved assets/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    d = load()
    spring, cluster = load("l16_saddle_eigen"), load("l16_cluster_check")
    check(spring, cluster)
    results = check_cluster(d)
    if not args.check_only:
        figures(d, results)
        for fig, name in [(spring_figure(spring), "L16-compressed-spring.png"),
                          (cluster["square_figure"], "L16-square-eigenvalues.png")]:
            fig.savefig(ASSETS / name, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved assets/{name}")
    plt.close("all")
    print("L16 numerical checks passed.")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9", "ase>=3.26", "weas-widget>=0.2.6"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from ase import Atoms
    from weas_widget import WeasWidget

    return Atoms, WeasWidget, minimize, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## L16 · Did the minimizer find a minimum? Four argon atoms from A1

    In Assignment 1, the tetrahedron was the lowest of three four-atom shapes. Here
    we start from a perfect **square** and let a minimizer relax it. The energy uses LJ
    reduced units ($\varepsilon=\sigma=1$), and the coordinates form a
    $4\times3$ matrix, so the Hessian is $12\times12$.
    """)
    return


@app.cell(hide_code=True)
def supplied_lj(np):
    def lj_energy(X):
        d = X[:, None, :] - X[None, :, :]
        r = np.linalg.norm(d, axis=-1)
        i, j = np.triu_indices(len(X), k=1)
        inv6 = r[i, j]**-6
        return np.sum(4*(inv6*inv6 - inv6))

    def lj_gradient(X):
        d = X[:, None, :] - X[None, :, :]
        r = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(r, np.inf)
        inv6 = r**-6
        return np.sum(((-48*inv6*inv6 + 24*inv6)/r**2)[:, :, None]*d, axis=1)

    return lj_energy, lj_gradient


@app.cell(hide_code=True)
def supplied_relax(lj_energy, lj_gradient, minimize):
    def relax(X0):
        N = len(X0)
        result = minimize(lambda x: lj_energy(x.reshape(N, 3)), X0.ravel(),
                          jac=lambda x: lj_gradient(x.reshape(N, 3)).ravel(),
                          method="BFGS", options={"gtol": 1e-8})
        return result.x.reshape(N, 3), result

    return (relax,)


@app.cell(hide_code=True)
def supplied_viewer(Atoms, WeasWidget, mo, np):
    def draw_clusters(structures, titles, sigma=3.40):
        panels = []
        for X, title in zip(structures, titles):
            atoms = Atoms("Ar"*len(X), positions=sigma*(X - X.mean(axis=0)))
            w = WeasWidget(from_ase=atoms, viewerStyle={"width": "100%", "height": "280px"},
                           modelStyle=1, cellSettings={"showCell": False},
                           cameraSetting={"lookAt": [0, 0, 0], "direction": [0.3, 0.4, 1.0],
                                          "distance": 25, "zoom": 1})
            w.avr.bond.add_bond_pair("Ar", max=1.3*2**(1/6)*sigma, color1="#888888", color2="#888888")
            panels.append(mo.vstack([mo.md(f"**{title}**"), mo.ui.anywidget(w.children[0])]))
        return mo.hstack(panels, widths="equal", align="start")

    return (draw_clusters,)


@app.cell(hide_code=True)
def relax_square(lj_energy, lj_gradient, np, relax):
    side = 2**(1/6)
    X_square0 = side*np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    X_square, square_result = relax(X_square0)
    E_square = lj_energy(X_square)
    square_gradient = np.abs(lj_gradient(X_square)).max()
    return E_square, X_square, square_gradient, square_result


@app.cell(hide_code=True)
def show_square(E_square, mo, square_gradient, square_result):
    mo.md(f"""
    ### The minimizer's report

    BFGS stops after {square_result.nit} iterations at **E = {E_square:.6f} ε** with
    `success = {square_result.success}` and largest gradient entry {square_gradient:.1e}.
    The structure is still a flat square. By symmetry, every force lies in the plane,
    so nothing pushes an atom out of it.
    """)
    return


@app.cell(hide_code=True)
def step_one_text(mo):
    mo.md(r"""
    ### Live step 1 · Hessian, eigenvalues, and a classification

    Build the $12\times12$ Hessian by central differences of the gradient, diagonalize
    it with `np.linalg.eigh`, and count eigenvalues that are negative, near zero
    ($\lvert\lambda\rvert<10^{-3}$), and positive. Six eigenvalues should be zero:
    three rigid translations and three rigid rotations.
    """)
    return


@app.cell
def live_classify(lj_gradient, np):
    # Live step 1: Hessian by central differences, its eigenpairs, and a count of each sign.
    def hessian(X, h=1e-5):
        x0 = X.ravel()
        H = np.empty((x0.size, x0.size))
        for j in range(x0.size):
            step = np.zeros(x0.size)
            step[j] = h
            H[:, j] = (lj_gradient((x0 + step).reshape(-1, 3)).ravel()
                       - lj_gradient((x0 - step).reshape(-1, 3)).ravel())/(2*h)
        return 0.5*(H + H.T)

    def classify(X, tol=1e-3):
        values, vectors = np.linalg.eigh(hessian(X))
        counts = {"negative": int(np.sum(values < -tol)),
                  "zero": int(np.sum(np.abs(values) <= tol)),
                  "positive": int(np.sum(values > tol))}
        return values, vectors, counts

    return (classify,)


@app.cell(hide_code=True)
def show_square_modes(X_square, classify, mo, np, plt):
    square_values, square_vectors, square_counts = classify(X_square)

    def draw():
        fig, ax = plt.subplots(figsize=(7, 3.2))
        colours = np.where(square_values < -1e-3, "tab:red",
                           np.where(np.abs(square_values) <= 1e-3, "tab:gray", "tab:blue"))
        ax.bar(np.arange(12), square_values, color=colours)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xlabel="Mode index (ascending)", ylabel="Eigenvalue (ε/σ²)",
               title="Hessian eigenvalues of the relaxed square")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    square_figure = draw()
    _z_share = [np.sum(square_vectors[:, k].reshape(4, 3)[:, 2]**2) for k in (0, 1)]
    mo.vstack([square_figure, mo.md(
        f"{square_counts['negative']} negative, {square_counts['zero']} zero, and "
        f"{square_counts['positive']} positive eigenvalues. Mode 1 (λ = {square_values[0]:.3f} ε/σ²) "
        f"keeps the atoms in the plane ({100*_z_share[0]:.0f}% out-of-plane): it shears the square "
        f"into a rhombus. Mode 2 (λ = {square_values[1]:.3f} ε/σ²) is {100*_z_share[1]:.0f}% "
        "out-of-plane: opposite corners rise while the other two sink.")])
    return square_counts, square_figure, square_values, square_vectors


@app.cell(hide_code=True)
def step_two_text(mo):
    mo.md(r"""
    ### Live step 2 · Follow a downhill eigenvector

    A negative eigenvalue gives a direction in which the energy falls. Displace every
    atom a small distance along that eigenvector and relax again. If the new structure
    still has a negative eigenvalue, repeat along its most negative mode.
    """)
    return


@app.cell
def live_follow(classify, relax):
    # Live step 2: push along an eigenvector (3N entries), relax, and repeat until no negative mode remains.
    def follow_mode(X, vector, step=0.1, max_steps=5):
        path = [X]
        for _ in range(max_steps):
            X, _result = relax(X + step*vector.reshape(-1, 3))
            path.append(X)
            values, vectors, counts = classify(X)
            if counts["negative"] == 0:
                break
            vector = vectors[:, 0]
        return path

    return (follow_mode,)


@app.cell(hide_code=True)
def mode_choice(mo):
    first_mode = mo.ui.radio({"Mode 1: in-plane shear": 0, "Mode 2: out-of-plane fold": 1},
                             value="Mode 1: in-plane shear", label="First mode to follow")
    first_mode
    return (first_mode,)


@app.cell(hide_code=True)
def show_follow(X_square, classify, draw_clusters, first_mode, follow_mode, lj_energy, mo, np, square_vectors):
    follow_path = follow_mode(X_square, square_vectors[:, first_mode.value])

    def describe(X):
        counts = classify(X)[2]
        r = np.sort(np.linalg.norm(X[:, None] - X[None], axis=-1)[np.triu_indices(4, 1)])
        kind = "minimum" if counts["negative"] == 0 else f"saddle ({counts['negative']} negative)"
        return f"| {lj_energy(X):.6f} | {kind} | {', '.join(f'{v:.3f}' for v in r)} |"

    X_folded = follow_path[-1]
    mo.vstack([
        mo.md("\n".join(["| Energy (ε) | Stationary point | Six pair distances (σ) |",
                         "|---:|---|---|", *[describe(X) for X in follow_path],
                         "",
                         "The final structure has all six distances equal to "
                         "$2^{1/6}\\sigma=1.122\\sigma$: the **tetrahedron** from A1."])),
        draw_clusters([X_square, X_folded], ["Relaxed square (a saddle)", "End of the path"]),
    ])
    return X_folded, follow_path


if __name__ == "__main__":
    app.run()

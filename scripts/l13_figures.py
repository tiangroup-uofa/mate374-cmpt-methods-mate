"""Static L13 figures for direct and iterative linear solvers.

Run from the repository root: uv run --locked python scripts/l13_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import scipy.linalg as sl

ASSETS = Path(__file__).resolve().parents[1] / "assets"
GREEN, ORANGE, GREY, BLUE, RED = "#007c41", "#d87700", "#9aa3ab", "#2f6fb0", "#c0392b"
DPI = 300


def draw_matrix(ax, x0, pattern, colors, label=None, size=0.5):
    """Draw an n x n grid; pattern holds keys into colors ('' is a white zero)."""
    n = len(pattern)
    for i in range(n):
        for j in range(n):
            key = pattern[i][j]
            face = colors.get(key, "white")
            ax.add_patch(Rectangle((x0 + j * size, -(i + 1) * size), size, size,
                                   facecolor=face, edgecolor="#555555", lw=0.8))
            if key == "1":
                ax.text(x0 + (j + 0.5) * size, -(i + 0.5) * size, "1",
                        ha="center", va="center", fontsize=10)
    if label:
        ax.text(x0 + n * size / 2, 0.25, label, ha="center", va="bottom", fontsize=13)
    return x0 + n * size


def vector(ax, x0, n, face, label, size=0.5):
    for i in range(n):
        ax.add_patch(Rectangle((x0, -(i + 1) * size), size, size,
                               facecolor=face, edgecolor="#555555", lw=0.8))
    ax.text(x0 + size / 2, 0.25, label, ha="center", va="bottom", fontsize=13)
    return x0 + size


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18,
                                 color="#333333", lw=1.5))


def direct_method_forms():
    n = 4
    full = [["a"] * n for _ in range(n)]
    upper = [["u" if j >= i else "" for j in range(n)] for i in range(n)]
    lower = [["1" if j == i else ("l" if j < i else "") for j in range(n)] for i in range(n)]
    ident = [["1" if j == i else "" for j in range(n)] for i in range(n)]
    colors = {"a": GREY, "u": GREEN, "l": ORANGE, "1": "#f3f3f3"}
    rows = [
        ("Gaussian elimination", [(upper, "U")], "x", "c"),
        ("LU factorization", [(lower, "L"), (upper, "U")], "x", "b"),
        ("Gauss–Jordan elimination", [(ident, "I")], "x", "c"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 4.6), layout="constrained")
    for ax, (name, factors, xname, rhs) in zip(axes, rows):
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.text(-0.3, -1.0, name, ha="right", va="center", fontsize=13, weight="bold")
        end = draw_matrix(ax, 0.0, full, colors, "A")
        end = vector(ax, end + 0.2, n, "white", "x")
        ax.text(end + 0.35, -1.0, "=", fontsize=16, ha="center", va="center")
        end = vector(ax, end + 0.7, n, "#dfe8f3", "b")
        arrow(ax, end + 0.3, end + 1.6, -1.0)
        x = end + 1.9
        for pattern, label in factors:
            x = draw_matrix(ax, x, pattern, colors, label) + 0.15
        x = vector(ax, x + 0.05, n, "white", xname)
        ax.text(x + 0.35, -1.0, "=", fontsize=16, ha="center", va="center")
        vector(ax, x + 0.7, n, "#dfe8f3" if rhs == "b" else "#cfe6da", rhs)
        ax.set_xlim(-5.4, 12.2)
        ax.set_ylim(-2.1, 0.7)
    fig.savefig(ASSETS / "L13-direct-method-forms.png", dpi=DPI)
    plt.close(fig)


def elimination_steps():
    n = 5
    fig, axes = plt.subplots(1, n, figsize=(11, 2.9), layout="constrained")
    for step, ax in enumerate(axes):
        ax.set_axis_off()
        ax.set_aspect("equal")
        last = step == n - 1
        for i in range(n):
            for j in range(n):
                if j < step and i > j:
                    face = "white"  # eliminated below a pivot
                elif (i < step or last) and j >= i:
                    face = GREEN  # finished row of U
                elif step > 0:
                    face = ORANGE  # trailing block updated by the last column
                else:
                    face = GREY
                ax.add_patch(Rectangle((j, -i - 1), 1, 1, facecolor=face,
                                       edgecolor="#555555", lw=0.8))
        if step < n - 1:
            ax.add_patch(Rectangle((step, -step - 1), 1, 1, facecolor="none",
                                   edgecolor="black", lw=2.6))
        title = "Start: A" if step == 0 else (f"After column {step}" if step < n - 1 else "Upper triangular U")
        ax.set_title(title, fontsize=12)
        if step > 0:
            ax.text(n / 2, -n - 0.55, f"{(n - step) ** 2} entr{'y' if step == n - 1 else 'ies'} updated",
                    ha="center", va="top", fontsize=10)
        ax.set_xlim(-0.1, n + 0.1)
        ax.set_ylim(-n - 1.3, 0.1)
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#555555") for c in (GREY, GREEN, ORANGE)]
    handles.append(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#555555"))
    handles.append(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", lw=2.6))
    fig.legend(handles, ["original entry", "finished row of U", "updated in this column",
                         "eliminated (zero)", "next pivot"],
               loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(ASSETS / "L13-elimination-steps.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def lu_reuse_cost():
    n = 1000
    m = np.unique(np.logspace(0, 3, 200).astype(int))
    factor, triangular = 2 * n**3 / 3, 2 * n**2
    repeat = m * (factor + triangular)
    reuse = factor + m * triangular
    inverse = 2 * n**3 + m * triangular
    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    ax.loglog(m, repeat, color=RED, lw=2.4, label="Eliminate again for every load")
    ax.loglog(m, inverse, color=GREY, lw=2.4, ls="--", label=r"Form $\mathbf{A}^{-1}$, then multiply")
    ax.loglog(m, reuse, color=GREEN, lw=2.4, label="Factor once, reuse L and U")
    ax.set(xlabel="Number of right-hand sides (load cases)",
           ylabel="Floating-point operations",
           title=f"Leading-order work for a dense system with n = {n}")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(ASSETS / "L13-lu-reuse-cost.png", dpi=DPI)
    plt.close(fig)


def eliminate_2x2(eps, pivot):
    A = np.array([[eps, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    if pivot and abs(A[1, 0]) > abs(A[0, 0]):
        A, b = A[::-1].copy(), b[::-1].copy()
    m = A[1, 0] / A[0, 0]
    u22 = A[1, 1] - m * A[0, 1]
    c2 = b[1] - m * b[0]
    x2 = c2 / u22
    x1 = (b[0] - A[0, 1] * x2) / A[0, 0]
    return np.array([x1, x2])


def pivoting_error():
    eps = np.logspace(-1, -19, 181)
    errors = {True: [], False: []}
    for e in eps:
        exact_x1 = 1 / (1 - e)
        for pivot in (True, False):
            errors[pivot].append(abs(eliminate_2x2(e, pivot)[0] - exact_x1) / abs(exact_x1))
    # Partial-pivoting errors are zero or one rounding unit; show them at that level.
    floor = np.finfo(float).eps
    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    ax.loglog(eps, np.maximum(errors[False], floor), color=RED, lw=2.4, label="No pivoting")
    ax.loglog(eps, np.maximum(errors[True], floor), color=GREEN, lw=2.4,
              label=r"Partial pivoting (error $\leq\epsilon_{\mathrm{mach}}$)")
    ax.loglog(eps, np.finfo(float).eps / eps, color=GREY, ls=":", lw=1.8,
              label=r"$\epsilon_{\mathrm{mach}}/\varepsilon$")
    ax.set(xlabel=r"Pivot size $\varepsilon$ in $a_{11}$",
           ylabel=r"Relative error in $x_1$", ylim=(1e-17, 1e3))
    ax.invert_xaxis()
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(ASSETS / "L13-pivoting-error.png", dpi=DPI)
    plt.close(fig)


def chain_matrix(n, k=5.0):
    K = k * (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    K[-1, -1] = k
    return K


def jacobi_history(A, b, sweeps):
    x = np.zeros_like(b)
    d = np.diag(A)
    xs, res = [x.copy()], [1.0]
    for _ in range(sweeps):
        x = x + (b - A @ x) / d
        xs.append(x.copy())
        res.append(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
    return np.array(xs), np.array(res)


def jacobi_spring_chain():
    n, k, F = 20, 5.0, 1.0
    K = chain_matrix(n, k)
    f = np.zeros(n)
    f[-1] = F
    xs, _ = jacobi_history(K, f, 2000)
    atoms = np.arange(n + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    left.plot(atoms, atoms * F / k, color="black", lw=2, ls="--", label="exact  $u_i = iF/k$")
    shades = plt.cm.YlGn(np.linspace(0.35, 0.95, 5))
    for color, sweep in zip(shades, (1, 10, 50, 200, 1000)):
        left.plot(atoms, np.r_[0.0, xs[sweep]], "o-", ms=3.5, color=color, lw=1.6,
                  label=f"after {sweep} sweep{'' if sweep == 1 else 's'}")
    left.set(xlabel="Atom index i (atom 0 fixed, load on atom 20)",
             ylabel="Displacement $u_i$ (Å)", title="Jacobi iterates, 20 moving atoms")
    left.legend(frameon=False, fontsize=9)
    left.grid(alpha=0.3)

    A3 = np.array([[4, -2, 1], [-2, 4, -2], [1, -2, 4.0]])
    b3 = np.array([11, -16, 17.0])
    _, r3 = jacobi_history(A3, b3, 3000)
    right.semilogy(r3, color=ORANGE, lw=2, label="3×3 worked example")
    for nn, color in [(5, BLUE), (20, GREEN), (80, GREY)]:
        Kn = chain_matrix(nn, k)
        fn = np.zeros(nn)
        fn[-1] = F
        _, rn = jacobi_history(Kn, fn, 3000)
        right.semilogy(rn, color=color, lw=2, label=f"spring chain, {nn} atoms")
    right.axhline(1e-6, color="black", ls=":", lw=1.2)
    right.text(3000, 2e-6, "tolerance $10^{-6}$", ha="right", fontsize=9)
    right.set(xlabel="Jacobi sweep", ylabel="Relative residual  $\\|b-Ax\\|/\\|b\\|$",
              ylim=(1e-12, 3), xlim=(0, 3000), title="Convergence depends on the matrix")
    right.legend(frameon=False, fontsize=9, loc="lower left")
    right.grid(alpha=0.3)
    fig.savefig(ASSETS / "L13-jacobi-spring-chain.png", dpi=DPI)
    plt.close(fig)


def lattice_fill_in():
    m = 12  # interior atoms per side; edge atoms are fixed
    T = 2 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)
    K = np.kron(np.eye(m), T) + np.kron(T, np.eye(m))
    L = sl.cholesky(K, lower=True)
    tol = 1e-12
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), layout="constrained",
                             gridspec_kw={"width_ratios": [0.9, 1, 1]})
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_axis_off()
    g = np.arange(m + 2)
    for i in g:
        ax.plot([0, m + 1], [i, i], color=GREY, lw=0.8, zorder=0)
        ax.plot([i, i], [0, m + 1], color=GREY, lw=0.8, zorder=0)
    X, Y = np.meshgrid(g, g)
    edge = (X == 0) | (Y == 0) | (X == m + 1) | (Y == m + 1)
    ax.scatter(X[edge], Y[edge], s=14, color="black", zorder=2, label="fixed atom")
    ax.scatter(X[~edge], Y[~edge], s=14, color=GREEN, zorder=2, label="moving atom")
    ax.set_title(f"Square lattice: {m}×{m} moving atoms", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    for ax, M, name in [(axes[1], K, "Stiffness matrix K"),
                        (axes[2], L + L.T, "Factors L and $L^{\\mathsf{T}}$")]:
        nz = np.abs(M) > tol
        ax.spy(nz, markersize=1.0, color=GREEN if M is K else ORANGE)
        ax.set_title(f"{name}\n{np.count_nonzero(nz)} nonzero entries", fontsize=11)
        ax.tick_params(labelsize=8)
    fig.savefig(ASSETS / "L13-fill-in.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 11})
    direct_method_forms()
    elimination_steps()
    lu_reuse_cost()
    pivoting_error()
    jacobi_spring_chain()
    lattice_fill_in()


if __name__ == "__main__":
    main()

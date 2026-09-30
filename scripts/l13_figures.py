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


def conditioning_error():
    """Relative error of np.linalg.solve against a prescribed condition number."""
    rng = np.random.default_rng(374)
    n = 6
    kappas, errors = [], []
    for kappa in np.logspace(0.5, 18.5, 37):
        for _ in range(8):
            # A = Q1 diag(s) Q2^T has singular values from 1 down to 1/kappa.
            Q1, _r = np.linalg.qr(rng.standard_normal((n, n)))
            Q2, _r = np.linalg.qr(rng.standard_normal((n, n)))
            A = Q1 @ np.diag(np.logspace(0, -np.log10(kappa), n)) @ Q2.T
            x_true = rng.standard_normal(n)
            try:
                x = np.linalg.solve(A, A @ x_true)
            except np.linalg.LinAlgError:
                continue
            kappas.append(kappa)
            errors.append(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))
    eps = np.finfo(float).eps
    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    ax.axvspan(1 / eps, 1e19, color=RED, alpha=0.08)
    ax.loglog(kappas, np.maximum(errors, 1e-17), "o", ms=3.5, color=GREEN, alpha=0.7,
              label="np.linalg.solve, random 6×6 matrices")
    k = np.logspace(0, 19, 50)
    ax.loglog(k, eps * k, color="black", ls=":", lw=2,
              label=r"rule of thumb: $\kappa\times\epsilon_{\mathrm{mach}}\approx\kappa\times10^{-16}$")
    ax.axhline(1, color=GREY, lw=1)
    ax.text(3e17, 1e-6, "no correct\ndigits left", color=RED, ha="center", fontsize=10)
    ax.set(xlabel=r"Condition number $\kappa_2(\mathbf{A})$",
           ylabel="Relative error in x", xlim=(1, 1e19), ylim=(1e-17, 1e3))
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.savefig(ASSETS / "L13-conditioning-error.png", dpi=DPI)
    plt.close(fig)


def solver_cheat_sheet():
    rows = [
        ("Start here", "Square, nonsingular A", "np.linalg.solve(A, b)",
         "LU with partial pivoting", r"$\frac{2}{3}n^3$"),
        ("Same A, loads arrive\none at a time?", "e.g. many load cases,\nrepeated steps",
         "lu = lu_factor(A)\nx = lu_solve(lu, b)", "factor once, reuse L, U",
         r"$\frac{2}{3}n^3$ once, $2n^2$ per b"),
        ("Symmetric positive\ndefinite?", "e.g. supported\nstiffness matrix K",
         "c = cho_factor(K)\nx = cho_solve(c, f)", r"Cholesky $\mathbf{K}=\mathbf{L}\mathbf{L}^{\mathsf{T}}$",
         r"$\frac{1}{3}n^3$, half the storage"),
        ("Nonzeros only near\nthe diagonal?", "e.g. nearest-neighbour\nchain",
         "solve_banded((p, p), ab, b)\nsolveh_banded(ab, b)  # SPD", "banded LU / Cholesky",
         r"about $np^2$; $n$ for tridiagonal"),
        ("Always check", "", "r = b - A @ x\nnp.linalg.cond(A)", "residual, conditioning",
         r"rel. error $\lesssim\kappa\times10^{-16}$"),
    ]
    heads = ["Question", "Typical case", "NumPy / SciPy call", "Method", "Leading work"]
    widths = [2.3, 2.1, 3.4, 2.4, 2.4]
    fills = ["#eef1f4", "#fdf3e6", "#e6f2ec", "#e6f2ec", "#f6f6f6"]
    fig, ax = plt.subplots(figsize=(12.6, 5.6), layout="constrained")
    ax.set_axis_off()
    x_edges = np.concatenate([[0], np.cumsum(widths)])
    h, head_h = 1.0, 0.55
    for j, text in enumerate(heads):
        ax.add_patch(Rectangle((x_edges[j], 0), widths[j], head_h, facecolor=GREEN,
                               edgecolor="white", lw=2))
        ax.text(x_edges[j] + widths[j] / 2, head_h / 2, text, ha="center", va="center",
                color="white", weight="bold", fontsize=11)
    for i, row in enumerate(rows):
        y = -(i + 1) * h
        for j, text in enumerate(row):
            face = fills[i] if j == 0 else ("#fbfbfb" if i % 2 else "white")
            ax.add_patch(Rectangle((x_edges[j], y), widths[j], h, facecolor=face,
                                   edgecolor="#c9ced3", lw=1))
            ax.text(x_edges[j] + 0.12 if j == 2 else x_edges[j] + widths[j] / 2, y + h / 2, text,
                    ha="left" if j == 2 else "center", va="center", fontsize=10.5,
                    family="monospace" if j == 2 else None,
                    weight="bold" if j == 0 else None)
        if 0 < i < len(rows) - 1:
            ax.annotate("", xy=(0.25, y + h - 0.02), xytext=(0.25, y + h + 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
    ax.text(0, -(len(rows) + 1) * h + 0.55,
            "Read top to bottom: each yes replaces the call above it. "
            "n = number of unknowns, p = number of nonzero diagonals on each side.",
            fontsize=10, color="#444444")
    ax.set_xlim(-0.05, x_edges[-1] + 0.05)
    ax.set_ylim(-(len(rows) + 1) * h + 0.3, head_h + 0.05)
    fig.savefig(ASSETS / "L13-solver-cheat-sheet.png", dpi=DPI)
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
    conditioning_error()
    solver_cheat_sheet()
    lattice_fill_in()


if __name__ == "__main__":
    main()

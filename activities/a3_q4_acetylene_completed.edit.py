# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A3 Q4 · Completed reference notebook

    The four-mass axial model has one rigid translation and three vibrational
    modes. The solved frequencies, displacement patterns, and measured IR
    comparison are summarized below.

    - Model: Linear H₁–C₂≡C₃–H₄ as four masses moving along one axis, with both ends free.
    - Spring constants: $k_{CH} = 592$ N/m (single bond C–H), $k_{CC} = 1580$ N/m (triple bond C≡C)
    - Masses: $m_H = 1$ a.u., $m_C = 12$ a.u.

    **Prediction:** What happens to the springs when all four atoms move by
    the same amount? How will that motion appear in the eigenvalues?

    ## Q4.1–4.2 · Eigenvalues and vectors

    Use `np.linalg.eig` to calculate the eigenvalues and eigenvectors of
    the dynamical matrix $\mathbf D$ defined in the assignment.

    $$D u=\omega^2 u.$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The matrix below is formed from the stated spring constants and masses.
    Its zero eigenvalue corresponds to rigid translation; the other three
    eigenvalues give the axial vibrational modes.
    """)
    return


@app.cell
def _(np):
    m_H, m_C = 1.0, 12.0 #a.u.
    k_CH = 592.0  # N/m
    k_CC = 1580.0  # N/m

    # Mass-divided stiffness matrix D from the assignment; no separate K is needed.
    D = np.array([
        [k_CH / m_H, -k_CH / m_H, 0, 0],
        [-k_CH / m_C, (k_CH + k_CC) / m_C, -k_CC / m_C, 0],
        [0, -k_CC / m_C, (k_CC + k_CH) / m_C, -k_CH / m_C],
        [0, 0, -k_CH / m_H, k_CH / m_H],
    ])
    return (D,)


@app.cell(hide_code=True)
def _(D, mo):
    mo.stop(
        D is ...,
        mo.callout("Q4.1: the matrix D has not been defined!", kind="danger"),
    )

    D_display = mo.ui.matrix(
        D,
        disabled=True,
        precision=3,
        label="Dynamical matrix D [N/(m·a.u.)]",
    )
    return (D_display,)


@app.cell(hide_code=True)
def _(D, D_display, mo):
    mo.stop(
        D is ...,
        mo.callout("Q4.1: the matrix D has not been defined!", kind="danger"),
    )
    D_display
    return


@app.cell
def _(D, np):
    # np.linalg.eig returns eigenvalues and matching eigenvectors, often as complex arrays.
    values_complex, vectors_complex = np.linalg.eig(D)
    if not np.allclose(values_complex.imag, 0, atol=1e-12):
        raise ValueError("This stable spring model should have real eigenvalues.")

    # Remove roundoff-sized imaginary parts, then sort eigenvalues from smallest to largest.
    values_real = values_complex.real
    vectors_real = vectors_complex.real
    order = np.argsort(values_real)
    eigenvalues_raw = values_real[order]
    displacements = vectors_real[:, order]
    zero_tolerance = 1e-10 * np.max(np.abs(eigenvalues_raw))
    if np.any(eigenvalues_raw < -zero_tolerance):
        raise ValueError("A significant negative eigenvalue indicates an unstable mode.")
    eigenvalues = np.where(np.abs(eigenvalues_raw) < zero_tolerance, 0.0, eigenvalues_raw)
    omega_model = np.sqrt(eigenvalues)
    displacements = displacements / np.max(np.abs(displacements), axis=0)
    # An eigenvector and its overall negative describe the same mode.
    displacements = displacements * np.where(displacements[0] < 0, -1, 1)
    print("Eigenvalues ω², in N/(m·a.u.):", eigenvalues)
    print("Angular frequencies ω, in √[N/(m·a.u.)]:", omega_model)
    print("Relative displacements (one mode per column):\n", displacements)
    return displacements, eigenvalues, omega_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q4.3 · Wavenumbers

    Convert the numerical angular frequencies above directly to wavenumbers:

    $$\nu=\frac{\omega}{2\pi\,(1.2216313\times10^{-3})}
    \quad(\mathrm{cm^{-1}}).$$

    Larger wavenumbers correspond to larger angular frequencies. Compare all four wavenumbers with the atomic motions.
    """)
    return


@app.cell
def _(np, omega_model):
    wavenumbers_cm = omega_model / (2 * np.pi * 1.2216313e-3)
    print("Wavenumbers (cm⁻¹):", wavenumbers_cm)
    return (wavenumbers_cm,)


@app.cell(hide_code=True)
def _(displacements, eigenvalues, mo, omega_model, wavenumbers_cm):
    mo.ui.table([
        {"Mode": _j + 1, "ω² (N/(m·a.u.))": round(float(eigenvalues[_j]), 6),
         "ω (√[N/(m·a.u.)])": round(float(omega_model[_j]), 6),
         "Wavenumber (cm⁻¹)": round(float(wavenumbers_cm[_j]), 3),
         "H1": round(float(displacements[0, _j]), 6),
         "C2": round(float(displacements[1, _j]), 6),
         "C3": round(float(displacements[2, _j]), 6),
         "H4": round(float(displacements[3, _j]), 6)}
        for _j in range(4)
    ], selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q4.3–4.4 · Relative atomic motions

    Select a mode and compare the arrow directions and lengths. At the instant
    shown, each atom is displaced in the direction of its arrow. A bond's change
    in length is the right atom's displacement minus the left atom's displacement.
    Check the bond changes printed beneath the plot to distinguish symmetric
    and asymmetric stretching.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    selected_mode = mo.ui.slider(
        start=1, stop=4, step=1, value=2,
        label="Mode, in increasing frequency order", show_value=True,
    )
    selected_mode
    return (selected_mode,)


@app.cell(hide_code=True)
def _(displacements, np, plt, selected_mode, wavenumbers_cm):
    _j = selected_mode.value - 1
    _a = displacements[:, _j]
    _x = np.arange(4, dtype=float)
    mode_figure, _ax = plt.subplots(figsize=(7, 2.2))
    _ax.plot(_x, np.zeros(4), color="#8a8f98", linewidth=1.2, zorder=1)
    _ax.scatter(_x, np.zeros(4), s=[350, 600, 600, 350],
                color=["white", "#e4e4e4", "#e4e4e4", "white"],
                edgecolors="#231f20", linewidths=1, zorder=2)
    for _i, _name in enumerate(["H₁", "C₂", "C₃", "H₄"]):
        _ax.text(_x[_i], 0, _name, ha="center", va="center", fontsize=12, zorder=3)
        _ax.annotate("", xy=(_x[_i] + 0.35 * _a[_i], -0.28),
                     xytext=(_x[_i], -0.28),
                     arrowprops={"arrowstyle": "-|>", "color": "#b5473a", "lw": 1.8})
        _ax.text(_x[_i], -0.49, f"{_a[_i]:+.3f}", ha="center", fontsize=11, color="#b5473a")
    _ax.set_title(f"Mode {_j + 1} · {wavenumbers_cm[_j]:.3f} cm⁻¹", fontsize=12)
    _ax.set_xlim(-0.55, 3.55)
    _ax.set_ylim(-0.65, 0.5)
    _ax.axis("off")
    mode_figure.tight_layout()
    mode_figure
    return


@app.cell
def _(displacements, np, selected_mode):
    _motion = displacements[:, selected_mode.value - 1]
    bond_changes = np.diff(_motion)
    print("Relative bond-length changes, left C–H / C≡C / right C–H:", bond_changes)
    print("Positive = extension; negative = contraction; zero = unchanged.")
    if np.allclose(_motion, _motion[0]):
        print("Rigid translation · no vibrational IR band.")
    elif np.allclose(_motion, -_motion[::-1]):
        print("Symmetric stretching · IR-inactive.")
    else:
        print("Asymmetric stretching · IR-active.")
    return


@app.cell
def _(D, displacements, eigenvalues, np):
    mode_residuals = D @ displacements - displacements * eigenvalues
    relative_residual = np.linalg.norm(mode_residuals) / (
        np.linalg.norm(D) * np.linalg.norm(displacements)
    )
    print("Relative eigenvalue residual:", relative_residual)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answers:** Mode 3, at approximately 3299.301 cm⁻¹, is the IR-active
    asymmetric C–H stretch associated with the measured band near 3258 cm⁻¹.
    The one-dimensional axial model cannot represent bending; allowing
    transverse atom displacements and adding angle-dependent restoring forces
    would introduce bending modes.
    """)
    return


if __name__ == "__main__":
    app.run()

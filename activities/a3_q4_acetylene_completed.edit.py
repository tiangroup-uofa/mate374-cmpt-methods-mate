# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
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

    Can we use a bead-and-spring model to explain the vibrational spectrum of molecules like C$_2$H$_2$ (acetylene)?

    - Model: linear H₁–C₂≡C₃–H₄ represented by four masses moving along one axis, with both ends free.
    - Spring constants: $k_{CH} = 592$ N/m (C–H single bond), $k_{CC} = 1580$ N/m (C≡C triple bond)
    - Masses: $m_H = 1$ a.u., $m_C = 12$ a.u.

    **Prediction:** What happens to the springs when all four atoms move by
    the same amount? How will that motion appear in the eigenvalues?

    ## Q4.1–4.2 · Eigenvalues and vectors

    The cells below use `np.linalg.eig` to solve this eigenvalue problem. The
    matrix $\mathbf D$ is defined in the assignment.

    $$D u=\omega^2 u.$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Q4.2:** the following cell writes the matrix `D` using `k_CH`, `k_CC`, `m_H`, and `m_C`.
    """)
    return


@app.cell
def _(np):
    m_H, m_C = 1.0, 12.0  # a.u.
    k_CH = 592.0  # N/m
    k_CC = 1580.0  # N/m

    # Explicit 4 × 4 dynamical matrix D from the assignment.
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
        mo.callout("Q4.2: replace the ellipsis with the matrix D.", kind="danger"),
    )
    mo.ui.matrix(
        D,
        disabled=True,
        precision=3,
        label="Dynamical matrix D [N/(m·a.u.)]",
    )
    return


@app.cell(hide_code=True)
def _(D, mo, np):
    # Compute the eigenvalues and matching eigenvectors of the dynamical matrix.
    # Each eigenvector is stored as a column; np.linalg.eig does not sort the modes.
    mo.stop(D is ...)
    eigenvalues_complex, eigenvectors_complex = np.linalg.eig(D)

    # This real spring model should have real eigenvalues; any imaginary parts should be roundoff.
    mo.stop(
        not np.allclose(eigenvalues_complex.imag, 0, atol=1e-12),
        mo.callout("The model produced a complex eigenvalue. Check the entries of D.", kind="danger"),
    )
    eigenvalues_real = eigenvalues_complex.real
    eigenvectors_real = eigenvectors_complex.real
    return eigenvalues_real, eigenvectors_real


@app.cell(hide_code=True)
def _(eigenvalues_real, eigenvectors_real, mo, np):
    # Sort eigenvalues from smallest to largest and apply the same ordering to eigenvector columns.
    _order = np.argsort(eigenvalues_real)
    _eigenvalues_sorted = eigenvalues_real[_order]
    _eigenvectors_sorted = eigenvectors_real[:, _order]

    # Treat roundoff-sized values as the zero-frequency translation mode.
    _zero_tolerance = 1e-10 * np.max(np.abs(_eigenvalues_sorted))
    mo.stop(
        np.any(_eigenvalues_sorted < -_zero_tolerance),
        mo.callout("A negative eigenvalue indicates an unstable spring model. Check D.", kind="danger"),
    )
    eigenvalues = np.where(np.abs(_eigenvalues_sorted) < _zero_tolerance, 0.0, _eigenvalues_sorted)
    omega_model = np.sqrt(eigenvalues)

    displacements = _eigenvectors_sorted / np.max(np.abs(_eigenvectors_sorted), axis=0)
    # An eigenvector and its overall negative describe the same mode.
    displacements = displacements * np.where(displacements[0] < 0, -1, 1)

    _table_rows = "\n".join(
        f"| {_j + 1} | {eigenvalues[_j]:.6f} | {omega_model[_j]:.6f} |"
        for _j in range(len(eigenvalues))
    )
    mo.md(
        "Eigenvalues and corresponding angular frequencies, ordered from smallest to largest:\n\n"
        "| Mode | Eigenvalue ω² [N/(m·a.u.)] | Angular frequency ω [√(N/(m·a.u.))] |\n"
        "|---:|---:|---:|\n"
        f"{_table_rows}"
    )
    return displacements, eigenvalues, omega_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q4.3 · Wavenumbers

    Convert each angular frequency to a wavenumber using

    $$\nu=\frac{\omega}{2\pi\,(1.2216313\times10^{-3})}
    \quad(\mathrm{cm^{-1}}).$$

    **Q4.3:** the following cell converts `omega_model` to wavenumbers. The final mode table shows the result.
    """)
    return


@app.cell
def _(np, omega_model):
    wavenumbers_cm = omega_model / (2 * np.pi * 1.2216313e-3)
    return (wavenumbers_cm,)


@app.cell(hide_code=True)
def _(displacements, eigenvalues, mo, omega_model, wavenumbers_cm):
    mo.ui.table([
        {
            "Mode": _j + 1,
            "ω² (N/(m·a.u.))": round(float(eigenvalues[_j]), 6),
            "ω (√[N/(m·a.u.)])": round(float(omega_model[_j]), 6),
            "Wavenumber (cm⁻¹)": round(float(wavenumbers_cm[_j]), 3),
            "u1": round(float(displacements[0, _j]), 6),
            "u2": round(float(displacements[1, _j]), 6),
            "u3": round(float(displacements[2, _j]), 6),
            "u4": round(float(displacements[3, _j]), 6),
        }
        for _j in range(4)
    ], selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q4.3–4.4 · Relative atomic motions

    The components $u_1$ through $u_4$ correspond to H₁, C₂, C₃, and H₄ in order. In the left schematic, each $u_i$ is labelled above its atom and a horizontal arrow shows its direction. The GIF on the right animates the same mode along the molecular axis, with its amplitude and timing scaled for display. Green marks IR-active motion; gray marks IR-inactive motion or rigid translation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    selected_mode = mo.ui.number(
        start=1, stop=4, step=1, value=2,
        label="Mode, in increasing frequency order",
    )
    selected_mode
    return (selected_mode,)


@app.cell(hide_code=True)
def _(
    activity_background,
    activity_color,
    mo,
    mode_activity,
    mode_index,
    mode_motion,
    mode_number,
    np,
    plt,
    wavenumbers_cm,
):
    import base64 as _base64
    from io import BytesIO as _BytesIO
    from PIL import Image as _Image, ImageDraw as _ImageDraw

    # Left: show each relative displacement component over its atom, with a directional arrow.
    mode_figure, _ax = plt.subplots(figsize=(7.2, 2.5))
    _atom_x = np.arange(4, dtype=float)
    _ax.plot(_atom_x, np.zeros(4), color="#8a8f98", linewidth=1.3, zorder=1)
    _ax.scatter(
        _atom_x,
        np.zeros(4),
        s=[360, 620, 620, 360],
        color=["white", "#e3e3e3", "#e3e3e3", "white"],
        edgecolors="#24272b",
        linewidths=1.2,
        zorder=2,
    )
    for _i, _atom_label in enumerate(["H₁", "C₂", "C₃", "H₄"]):
        _ax.text(_atom_x[_i], 0, _atom_label, ha="center", va="center", fontsize=11, zorder=3)
        _ax.annotate(
            "",
            xy=(_atom_x[_i] + 0.38 * mode_motion[_i], 0.31),
            xytext=(_atom_x[_i], 0.31),
            arrowprops={"arrowstyle": "-|>", "color": activity_color, "lw": 2.0, "mutation_scale": 12},
            zorder=4,
        )
        _ax.text(
            _atom_x[_i], 0.66, f"u{_i + 1} = {mode_motion[_i]:+.3f}",
            ha="center", va="center", fontsize=10, fontweight="bold", color=activity_color,
        )
    _ax.set_title("Relative displacement components", fontsize=11, pad=9)
    _ax.set_xlim(-0.55, 3.55)
    _ax.set_ylim(-0.34, 0.9)
    _ax.axis("off")
    mode_figure.tight_layout()

    # Right: create an animated GIF for the selected eigenvector.
    _image_size = (480, 180)
    _base_positions = np.linspace(65, 415, 4)
    _atom_radii = [16, 24, 24, 16]
    _atom_names = ["H1", "C2", "C3", "H4"]
    _frames = []
    for _phase in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        _frame = _Image.new("RGB", _image_size, "white")
        _draw = _ImageDraw.Draw(_frame)
        _positions = _base_positions + 0.22 * (_base_positions[1] - _base_positions[0]) * mode_motion * np.sin(_phase)
        _center_y = 91
        _draw.text((18, 16), f"Mode {mode_number} · axial motion", fill="#454a52")

        # Draw one C–H spring and three parallel lines for the C≡C spring.
        for _bond_index in range(3):
            _left = _positions[_bond_index] + _atom_radii[_bond_index]
            _right = _positions[_bond_index + 1] - _atom_radii[_bond_index + 1]
            if _bond_index == 1:
                for _offset in (-5, 0, 5):
                    _draw.line((_left, _center_y + _offset, _right, _center_y + _offset), fill=activity_color, width=2)
            else:
                _draw.line((_left, _center_y, _right, _center_y), fill=activity_color, width=2)

        for _atom_index, _atom_name in enumerate(_atom_names):
            _x = int(round(_positions[_atom_index]))
            _radius = _atom_radii[_atom_index]
            _fill = "white" if _atom_index in (0, 3) else "#e3e3e3"
            _draw.ellipse(
                (_x - _radius, _center_y - _radius, _x + _radius, _center_y + _radius),
                fill=_fill,
                outline="#24272b",
                width=2,
            )
            _draw.text((_x, _center_y), _atom_name, fill="#24272b", anchor="mm")
        _frames.append(_frame)

    _gif_buffer = _BytesIO()
    _frames[0].save(
        _gif_buffer,
        format="GIF",
        save_all=True,
        append_images=_frames[1:],
        duration=90,
        loop=0,
        disposal=2,
    )
    _gif_data = _base64.b64encode(_gif_buffer.getvalue()).decode("ascii")
    _mode_gif = mo.Html(
        f'<img src="data:image/gif;base64,{_gif_data}" '
        f'alt="Animated axial motion for mode {mode_number}" '
        'style="display:block;width:100%;max-width:500px;height:auto;border:1px solid #d4d6da;border-radius:8px;" />'
    )

    _activity_badge = mo.Html(
        f'<div role="status" style="display:inline-block;padding:0.45rem 0.75rem;'
        f'border-radius:6px;background:{activity_background};color:{activity_color};'
        f'font-weight:600;">{mode_activity}</div>'
    )
    mo.vstack(
        [
            mo.md(f"### Mode {mode_number} · {wavenumbers_cm[mode_index]:.1f} cm⁻¹"),
            mo.hstack(
                [
                    mo.vstack([mode_figure]),
                    mo.vstack([mo.md("**Molecular motion**"), _mode_gif]),
                ],
                justify="start",
                align="start",
                widths=[1.0, 1.15],
                gap=1,
                wrap=True,
            ),
            _activity_badge,
        ]
    )
    return


@app.cell(hide_code=True)
def _(displacements, eigenvalues, mo, np, selected_mode, wavenumbers_cm):
    _mode_value = float(selected_mode.value)
    mo.stop(
        not np.isclose(_mode_value, round(_mode_value)),
        mo.callout("Choose a whole-number mode from 1 to 4.", kind="warn"),
    )
    mode_number = int(round(_mode_value))
    mo.stop(
        not 1 <= mode_number <= len(eigenvalues),
        mo.callout("Choose a whole-number mode from 1 to 4.", kind="warn"),
    )
    mode_index = mode_number - 1
    mode_motion = displacements[:, mode_index]

    if np.allclose(mode_motion, mode_motion[0]):
        mode_activity = "Rigid translation · no vibrational IR band"
        ir_active = False
        activity_color = "#666b73"
        activity_background = "#f1f2f4"
    elif np.allclose(mode_motion, -mode_motion[::-1]):
        mode_activity = "IR inactive · symmetric stretch"
        ir_active = False
        activity_color = "#666b73"
        activity_background = "#f1f2f4"
    else:
        mode_activity = "IR active · asymmetric stretch"
        ir_active = True
        activity_color = "#287a3d"
        activity_background = "#e8f5eb"
    None
    return (
        activity_background,
        activity_color,
        mode_activity,
        mode_index,
        mode_motion,
        mode_number,
    )


@app.cell(hide_code=True)
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
    **Spectrum comparison (Q4.4):** Mode 3, at about 3299.3 cm⁻¹, is the
    asymmetric C–H stretch and the only IR-active mode. It explains the
    measured band near 3258 cm⁻¹. One C–H bond contracts while the other
    extends, so the two C–H bond dipoles stop cancelling and the net dipole
    oscillates.

    **Model extension (Q4.5):** the atoms should be able to move in all three
    directions, so bending and wiggling motions are also vibrations. The
    strongest band, near 730 cm⁻¹, may be explained by bending. A linear
    molecule with $N$ atoms has $3N-5$ vibrational modes, so C₂H₂ has 7.
    """)
    return


if __name__ == "__main__":
    app.run()

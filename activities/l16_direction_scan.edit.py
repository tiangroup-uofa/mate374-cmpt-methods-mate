# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    return TwoSlopeNorm, mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    **Which direction lowers the energy most for the same displacement?**
    Rotate the blue unit direction in 5° steps. Predict the direction from the
    compressed springs before looking for the lowest energy cut.

    The compressed spring model is now simplified near the saddle point, where
    $E(x,y)-E(0,0)=2x^2+6xy+2y^2$, with
    $\mathbf H=\begin{bmatrix}4&6\\6&4\end{bmatrix}$.
    """)
    return


@app.cell(hide_code=True)
def control(mo):
    angle = mo.ui.slider(start=0, stop=355, step=5, value=0,
                         label="Direction θ (degrees)", show_value=True, debounce=True)
    angle
    return (angle,)


@app.cell
def model(np):
    H = np.array([[4.0, 6.0], [6.0, 4.0]])

    def energy(x, y):
        return 2*x**2 + 6*x*y + 2*y**2

    def scan_direction(degrees):
        theta = np.deg2rad(degrees)
        u = np.array([np.cos(theta), np.sin(theta)])
        curvature = float(u @ H @ u)
        transverse = H @ u - curvature*u
        aligned = np.linalg.norm(transverse) < 1e-10
        return u, curvature, aligned

    return H, energy, scan_direction


@app.cell(hide_code=True)
def drawing(H, TwoSlopeNorm, energy, np, plt, scan_direction):
    def draw_direction(degrees):
        u, curvature, aligned = scan_direction(degrees)
        q = np.linspace(-0.2, 0.2, 201)
        points = q[:, None]*u
        cut_energy = energy(points[:, 0], points[:, 1])
        probe = 0.1*u
        gradient = H @ probe
        # Only the gradient arrow is normalized for display.
        gradient_arrow = 0.075*gradient/np.linalg.norm(gradient)
        grid = np.linspace(-0.22, 0.22, 151)
        X, Y = np.meshgrid(grid, grid)
        Z = energy(X, Y)

        if aligned and curvature < 0:
            verdict = "Found: most downhill direction"
        elif aligned:
            verdict = "Eigenvector found: uphill"
        elif curvature < 0:
            verdict = "Downhill cut; keep rotating"
        else:
            verdict = "Uphill cut; keep rotating"

        with plt.rc_context({"font.family": "Arial", "font.size": 12,
                             "axes.titlesize": 13}):
            fig, (map_ax, cut_ax) = plt.subplots(
                1, 2, figsize=(10.6, 4.5), layout="constrained"
            )
            field = map_ax.pcolormesh(
                X, Y, Z, shading="auto", cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.5),
                rasterized=True,
            )
            map_ax.contour(X, Y, Z, levels=[-0.08, -0.04, 0, 0.04, 0.1, 0.2, 0.4],
                           colors="#5e5e5e", linewidths=0.5, alpha=0.7)
            fig.colorbar(field, ax=map_ax, shrink=0.72, label="ΔE")
            map_ax.plot(points[:, 0], points[:, 1], "--", color="#1f77b4", lw=1.5)
            map_ax.quiver(0, 0, *probe, angles="xy", scale_units="xy", scale=1,
                          color="#1f77b4", width=0.009, label="u direction")
            map_ax.quiver(*probe, *gradient_arrow, angles="xy", scale_units="xy",
                          scale=1, color="#231f20", width=0.009,
                          label=r"$\nabla E$ at q = 0.10")
            map_ax.plot(0, 0, "o", color="#231f20", ms=5)
            map_ax.plot(*probe, "o", color="#1f77b4", ms=5)
            map_ax.set(xlim=(-0.22, 0.22), ylim=(-0.22, 0.22),
                       xlabel="x", ylabel="y", aspect="equal",
                       title=f"Direction through (0, 0): {degrees}°")
            map_ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

            cut_ax.axhline(0, color="#8a8f98", lw=0.8)
            cut_ax.axvline(0, color="#8a8f98", lw=0.8, ls=":")
            cut_ax.plot(q, cut_energy, color="#b5473a", lw=2.5)
            cut_ax.plot(0, 0, "o", color="#231f20", ms=5)
            cut_ax.plot(0.1, energy(*probe), "o", color="#1f77b4", ms=6)
            cut_ax.set(xlim=(-0.2, 0.2), ylim=(-0.05, 0.22),
                       xlabel="Signed displacement q along ±u", ylabel="ΔE",
                       title="Energy along the selected line")
            cut_ax.text(0.5, 0.94, verdict, transform=cut_ax.transAxes,
                        ha="center", va="top", fontsize=11, fontweight="bold",
                        color="#2ca02c" if aligned else "#b5473a")
            cut_ax.text(0.5, 0.84,
                        rf"$\mathbf{{u}}^T\mathbf{{H}}\mathbf{{u}} = {curvature:.3f}$",
                        transform=cut_ax.transAxes, ha="center", va="top")
            cut_ax.grid(alpha=0.15)
        return fig

    return (draw_direction,)


@app.cell(hide_code=True)
def display(angle, draw_direction, mo, plt):
    figure = draw_direction(angle.value)
    mo.vstack([
        figure,
        mo.md(r"""
        The blue point marks $q=0.10$ on both plots. The black arrow shows the
        **gradient direction there**, scaled to a fixed length for visibility.
        At the origin itself, $\nabla E=\mathbf 0$.
        """),
    ])
    return (figure,)


if __name__ == "__main__":
    app.run()

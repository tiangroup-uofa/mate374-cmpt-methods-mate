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

    return mo, np, plt


@app.cell(hide_code=True)
def controls(mo):
    surface_choice = mo.ui.radio(
        options={
            "Saddle: E = 2x² − y²": "saddle",
            "Minimum: E = 2x² + y²": "minimum",
        },
        value="Saddle: E = 2x² − y²", label="Energy surface", inline=True,
    )
    return (surface_choice,)


@app.cell(hide_code=True)
def surfaces(np, surface_choice):
    # Both surfaces have a stationary point at the origin and a diagonal Hessian.
    H_xx = 4.0
    H_yy = -2.0 if surface_choice.value == "saddle" else 2.0
    H = np.array([[H_xx, 0.0], [0.0, H_yy]])

    def energy(x, y):
        return 0.5*(H_xx*x**2 + H_yy*y**2)

    return H, H_xx, H_yy, energy


@app.cell(hide_code=True)
def plot_surface(H_xx, H_yy, energy, mo, np, plt, surface_choice):
    def draw():
        s = np.linspace(-1.0, 1.0, 61)
        X, Y = np.meshgrid(s, s)
        Z = energy(X, Y)
        with plt.rc_context({"font.size": 11, "axes.titlesize": 11}):
            fig = plt.figure(figsize=(9.6, 3.9))
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0,
                            rstride=2, cstride=2)
            ax.plot(s, 0*s, energy(s, 0*s), color="#1f77b4", lw=2.5, zorder=5)
            ax.plot(0*s, s, energy(0*s, s), color="#b5473a", lw=2.5, zorder=5)
            ax.scatter([0], [0], [0], color="#ffcf40", edgecolor="#231f20", s=60, zorder=6)
            ax.set(xlabel="Δx", ylabel="Δy", zlabel="ΔE", title="Energy surface")
            ax.view_init(elev=24, azim=-58)

            cut = fig.add_subplot(1, 2, 2)
            cut.plot(s, energy(s, 0*s), color="#1f77b4", lw=2,
                     label=f"along x:  ½·({H_xx:+.0f})·Δx²")
            cut.plot(s, energy(0*s, s), color="#b5473a", lw=2,
                     label=f"along y:  ½·({H_yy:+.0f})·Δy²")
            cut.axhline(0, color="#8a8f98", lw=0.6)
            cut.axvline(0, color="#8a8f98", lw=0.6, ls=":")
            cut.plot(0, 0, "o", color="#ffcf40", mec="#231f20", ms=8, zorder=5)
            cut.set(xlabel="Displacement from the stationary point", ylabel="ΔE",
                    title="Energy along the two coordinate axes")
            cut.legend(fontsize=9, loc="upper center")
            cut.grid(alpha=0.2)
            fig.tight_layout()
        return fig

    surface_figure = draw()
    mo.vstack([surface_choice, surface_figure])
    return (surface_figure,)


@app.cell(hide_code=True)
def show_hessian(H_xx, H_yy, mo, surface_choice):
    verdict = ("both curvatures are positive, so every small displacement raises the energy: "
               "a **local minimum**"
               if surface_choice.value == "minimum" else
               "the curvatures have opposite signs, so the energy rises along $x$ and falls "
               "along $y$: a **saddle point**")
    mo.md(rf"""
    $$
    \mathbf H=\begin{{bmatrix}}{H_xx:+.0f} & 0\\ 0 & {H_yy:+.0f}\end{{bmatrix}},
    \qquad
    \Delta E\approx\tfrac12\left(H_{{xx}}\,\Delta x^2+H_{{yy}}\,\Delta y^2\right)
    =\tfrac12\left(({H_xx:+.0f})\,\Delta x^2+({H_yy:+.0f})\,\Delta y^2\right).
    $$

    The off-diagonal entries are zero, so each diagonal entry is the curvature of one
    coloured cut. Here {verdict}.
    """)
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "matplotlib>=3.11"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # A2 Q2 · Fixed points and phase compositions

    **Materials question:** when does the symmetric model have two distinct
    equilibrium compositions, and can fixed-point iteration reach one of them?

    Let $x$ be the atomic fraction of A in the A–B alloy, with $0<x<1$.
    The dimensionless parameter $H$ measures the energetic penalty for mixing
    relative to the thermal contribution. The model equation is

    $$
    F(x;H)=\ln\frac{x}{1-x}+H(1-2x)=0.
    $$

    Before using the controls, predict how the intersections change as $H$
    increases. The fixed-point map should preserve the central solution
    $x=0.5$; for sufficiently large $H$, two outer fixed points also appear.
    """)
    return


@app.cell(hide_code=True)
def rearrange_equation(mo):
    mo.md(r"""
    ## Rearrange the equation

    Isolate the logarithm, exponentiate, and solve for $x$:

    $$
    \begin{aligned}
    \ln\frac{x}{1-x} &= H(2x-1),\\
    \frac{x}{1-x} &= \exp\!\left[H(2x-1)\right],\\
    x &= g(x;H)=\frac{1}{1+\exp\!\left[H(1-2x)\right]}.
    \end{aligned}
    $$

    Edit `g(x, H)` below if your rearrangement differs. The plot compares
    $y=g(x;H)$ with $y=x$. Their intersections are fixed points of the map.
    """)
    return


@app.cell(hide_code=False)
def student_map(np):
    def g(x, H):
        # Edit this function to test your rearrangement.
        return 1.0 / (1.0 + np.exp(H * (1.0 - 2.0 * x)))

    return (g,)


@app.cell(hide_code=True)
def model_residual(np):
    def F(x, H):
        return np.log(x / (1.0 - x)) + H * (1.0 - 2.0 * x)

    return (F,)


@app.cell(hide_code=True)
def controls(mo):
    strength = mo.ui.slider(
        0.5, 3.0, step=0.1, value=2.5,
        label="Dimensionless mixing parameter H (try 1.5, 2.0, and 2.5)",
        show_value=True,
    )
    start = mo.ui.number(
        0.01, 0.99, value=0.9, step=0.01,
        label="Starting composition x₀ (use 0.90 for Q2.4)",
    )
    mo.vstack([strength, start])
    return strength, start


@app.cell(hide_code=True)
def plot_intersections(F, g, mo, np, plt, strength):
    _H = float(strength.value)
    x_values = np.linspace(0.0, 1.0, 1001)

    try:
        g_values = np.array([float(g(float(x), _H)) for x in x_values])
        if not np.all(np.isfinite(g_values)):
            raise ValueError("`g(x, H)` returned a non-finite value.")
    except Exception as error:
        plot_output = mo.callout(
            mo.md(f"**The plot could not be made.** {error} Check the editable `g(x, H)` function."),
            kind="warn",
        )
    else:
        difference = g_values - x_values
        intersections = []
        for index in range(len(x_values) - 1):
            left, right = difference[index], difference[index + 1]
            if left == 0.0:
                intersections.append(float(x_values[index]))
            elif left * right < 0.0:
                fraction = -left / (right - left)
                intersections.append(float(x_values[index] + fraction * (x_values[index + 1] - x_values[index])))
        if difference[-1] == 0.0:
            intersections.append(float(x_values[-1]))

        fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
        ax.plot(x_values, g_values, color="#007c41", lw=2.2, label=r"$y=g(x;H)$")
        ax.plot(x_values, x_values, color="0.35", ls="--", label=r"$y=x$")
        ax.scatter([0.5], [0.5], marker="o", s=62, facecolors="none", edgecolors="#bb3e03",
                   linewidths=1.8, zorder=4, label=r"Expected central point $(0.5,0.5)$")
        if intersections:
            ax.scatter(intersections, intersections, color="#bb3e03", s=24, zorder=5,
                       label="Sampled intersections")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Composition x", ylabel="Map output y",
               title=f"Fixed points for H = {_H:g}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8, loc="upper left")

        _rows = [
            {"Approximate fixed point x": root, "Original residual F(x; H)": float(F(root, _H))}
            for root in intersections if 0.0 < root < 1.0
        ]
        center_value = float(g(0.5, _H))
        if abs(center_value - 0.5) < 1e-8:
            center_check = mo.callout(
                mo.md(f"At the centre, `g(0.5, H) = {center_value:.8f}`. The map fixes $x=0.5$."),
                kind="success",
            )
        else:
            center_check = mo.callout(
                mo.md(f"At the centre, `g(0.5, H) = {center_value:.8f}`, not 0.5. Check the rearrangement."),
                kind="warn",
            )

        if _rows:
            result_table = mo.ui.table(_rows, selection=None)
        else:
            result_table = mo.md("No fixed-point intersections were detected on the plotted interval.")
        plot_output = mo.vstack([
            fig,
            center_check,
            mo.md("**Approximate intersections** (the residual column checks each one in the original equation):"),
            result_table,
        ])
    plot_output
    return


@app.cell(hide_code=True)
def iteration_instructions(mo):
    mo.md(r"""
    ## Q2.4 · Iterate to the upper composition

    Set $H=2.5$ and $x_0=0.90$. Each row below records one update
    $x_{n+1}=g(x_n;H)$ and checks the new value in the original equation.
    The iteration stops when

    $$
    |x_{n+1}-x_n|<10^{-4}.
    $$

    Compare the update count, final composition, and residual with your own
    calculation. A small step is the requested stopping test; the residual
    provides a separate check that the returned value solves $F(x;H)=0$.
    """)
    return


@app.cell(hide_code=True)
def trace_iteration(F, g, mo, np, start, strength):
    _H = float(strength.value)
    x = float(start.value)
    tolerance = 1e-4
    max_updates = 100
    _rows = []

    if not np.isfinite(x) or not 0.0 < x < 1.0:
        iteration_output = mo.callout(
            mo.md("Enter a finite starting composition strictly between 0 and 1."), kind="warn"
        )
    else:
        _rows.append({"Update n": 0, "Composition xₙ": x, "Step |xₙ − xₙ₋₁|": None,
                      "Original residual F(xₙ; H)": float(F(x, _H))})
        status = "Iteration limit reached before the step tolerance was met."
        converged = False
        error_message = None

        for update in range(1, max_updates + 1):
            try:
                next_x = float(g(x, _H))
            except Exception as error:
                error_message = f"The map failed at update {update}: {error}"
                break
            if not np.isfinite(next_x) or not 0.0 < next_x < 1.0:
                error_message = f"The map returned x = {next_x!r}, outside the composition domain 0 < x < 1."
                break

            step = abs(next_x - x)
            residual = float(F(next_x, _H))
            _rows.append({"Update n": update, "Composition xₙ": next_x,
                          "Step |xₙ − xₙ₋₁|": step, "Original residual F(xₙ; H)": residual})
            x = next_x
            if step < tolerance:
                status = f"Step tolerance met after {update} updates."
                converged = True
                break

        if error_message:
            iteration_result = mo.callout(mo.md(f"**Iteration stopped.** {error_message}"), kind="warn")
        elif converged:
            composition_name = "upper composition xβ" if _H > 2.0 and x > 0.5 else "fixed point"
            iteration_result = mo.vstack([
                mo.callout(mo.md(f"**{status}**"), kind="success"),
                mo.md(
                    f"Final {composition_name}: **{x:.5g}**  \n"
                    f"Updates: **{len(_rows) - 1}**  \n"
                    f"Original residual: **F(x; H) = {F(x, _H):+.3e}**"
                ),
            ])
        else:
            iteration_result = mo.callout(mo.md(f"**{status}**"), kind="warn")
        iteration_output = mo.vstack([iteration_result, mo.ui.table(_rows, selection=None)])
    iteration_output
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Coupled equilibrium and the Jacobian

    Now imagine two thermally coupled surface regions with temperatures
    $T_1$ and $T_2$. Each region loses heat by convection and radiation, and
    the regions exchange heat through a conductive coupling:

    $$
    \begin{aligned}
    F_1(T_1,T_2) &= h(T_1-T_\infty)
    +\varepsilon\sigma(T_1^4-T_\infty^4)+k(T_1-T_2)-q_1'',\\
    F_2(T_1,T_2) &= h(T_2-T_\infty)
    +\varepsilon\sigma(T_2^4-T_\infty^4)+k(T_2-T_1)-q_2''.
    \end{aligned}
    $$

    Equilibrium requires **both** residuals to vanish:
    $\mathbf{F}(\mathbf{T})=\mathbf{0}$. Newton's method linearizes the
    coupled problem and solves

    $$
    J(\mathbf{T}_k)\,\Delta\mathbf{T}=-\mathbf{F}(\mathbf{T}_k),
    \qquad \mathbf{T}_{k+1}=\mathbf{T}_k+\Delta\mathbf{T}.
    $$

    This is our first appearance of `Ax=b` as an inner problem inside a
    nonlinear materials calculation.
    """)
    return


@app.cell
def _(mo):
    heat_flux_1 = mo.ui.slider(
        start=2_000,
        stop=20_000,
        step=1_000,
        value=10_000,
        label="Region 1 heat flux, q₁″ (W/m²)",
        show_value=True,
    )
    heat_flux_2 = mo.ui.slider(
        start=2_000,
        stop=20_000,
        step=1_000,
        value=5_000,
        label="Region 2 heat flux, q₂″ (W/m²)",
        show_value=True,
    )
    initial_temperature = mo.ui.slider(
        start=350,
        stop=1_000,
        step=25,
        value=500,
        label="Initial guess for both temperatures (K)",
        show_value=True,
    )
    mo.vstack([heat_flux_1, heat_flux_2, initial_temperature], align="start", gap=1)
    return heat_flux_1, heat_flux_2, initial_temperature


@app.cell
def _(np):
    STEFAN_BOLTZMANN = 5.670374419e-8
    CONVECTION_COEFFICIENT = 15.0
    AMBIENT_TEMPERATURE = 300.0
    EMISSIVITY = 0.80
    COUPLING_CONDUCTANCE = 20.0

    def coupled_residual(temperatures, heat_fluxes):
        # Given model: edit the body only after identifying every term.
        t1, t2 = temperatures
        q1, q2 = heat_fluxes
        common_1 = CONVECTION_COEFFICIENT * (t1 - AMBIENT_TEMPERATURE)
        common_2 = CONVECTION_COEFFICIENT * (t2 - AMBIENT_TEMPERATURE)
        radiation_1 = EMISSIVITY * STEFAN_BOLTZMANN * (
            t1**4 - AMBIENT_TEMPERATURE**4
        )
        radiation_2 = EMISSIVITY * STEFAN_BOLTZMANN * (
            t2**4 - AMBIENT_TEMPERATURE**4
        )
        return np.array(
            [
                common_1 + radiation_1 + COUPLING_CONDUCTANCE * (t1 - t2) - q1,
                common_2 + radiation_2 + COUPLING_CONDUCTANCE * (t2 - t1) - q2,
            ]
        )

    def coupled_jacobian(temperatures):
        t1, t2 = temperatures
        diagonal_1 = (
            CONVECTION_COEFFICIENT
            + 4.0 * EMISSIVITY * STEFAN_BOLTZMANN * t1**3
            + COUPLING_CONDUCTANCE
        )
        diagonal_2 = (
            CONVECTION_COEFFICIENT
            + 4.0 * EMISSIVITY * STEFAN_BOLTZMANN * t2**3
            + COUPLING_CONDUCTANCE
        )
        return np.array(
            [
                [diagonal_1, -COUPLING_CONDUCTANCE],
                [-COUPLING_CONDUCTANCE, diagonal_2],
            ]
        )

    return coupled_jacobian, coupled_residual


@app.cell
def _(np):
    def newton_system(residual, jacobian, initial_guess, tolerance, max_iterations):
        # The linear solve is the step students should inspect.
        current = np.array(initial_guess, dtype=float)
        history = []
        for iteration in range(1, max_iterations + 1):
            residual_value = residual(current)
            residual_norm = np.linalg.norm(residual_value, ord=np.inf)
            history.append(
                (iteration, current.copy(), residual_norm, 0.0)
            )
            if residual_norm <= tolerance:
                return current, history, "converged"
            jacobian_value = jacobian(current)
            step = np.linalg.solve(jacobian_value, -residual_value)
            current = current + step
            history[-1] = (
                iteration,
                history[-1][1],
                residual_norm,
                np.linalg.norm(step, ord=np.inf),
            )
        return current, history, "iteration limit"

    return (newton_system,)


@app.cell
def _(
    coupled_jacobian,
    coupled_residual,
    heat_flux_1,
    heat_flux_2,
    initial_temperature,
    newton_system,
    np,
):
    heat_fluxes = np.array([heat_flux_1.value, heat_flux_2.value], dtype=float)

    def residual(temperatures):
        return coupled_residual(temperatures, heat_fluxes)

    root, history, status = newton_system(
        residual,
        coupled_jacobian,
        [initial_temperature.value, initial_temperature.value],
        tolerance=1.0e-3,
        max_iterations=30,
    )
    final_residual = residual(root)
    return final_residual, history, root, status


@app.cell
def _(final_residual, history, mo, root, status):
    rows = "\n".join(
        f"| {iteration} | {temperatures[0]:.3f} | {temperatures[1]:.3f} | {residual_norm:.3e} | {step_norm:.3e} |"
        for iteration, temperatures, residual_norm, step_norm in history
    )
    result = mo.callout(
        mo.md(
            fr"""
            ## Coupled Newton result

            - status: **{status}**
            - region 1 temperature: **$T_1={root[0]:.3f}\,\mathrm{{K}}$**
            - region 2 temperature: **$T_2={root[1]:.3f}\,\mathrm{{K}}$**
            - final residual vector: **$({final_residual[0]:.2e},\,{final_residual[1]:.2e})\,\mathrm{{W/m^2}}$**

            | Iteration | $T_1$ (K) | $T_2$ (K) | $\|F\|_\infty$ | $\|\Delta T\|_\infty$ |
            |---:|---:|---:|---:|---:|
            {rows}
            """
        ),
        kind="success" if status == "converged" else "warn",
    )
    result
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### What to inspect

            The Newton update is not a mysterious multivariable formula: it
            assembles a Jacobian, solves a small linear system, and updates the
            state. Change the initial guess and ask whether failure came from
            the physical model, the Jacobian, the linear solve, or the local
            convergence assumption.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

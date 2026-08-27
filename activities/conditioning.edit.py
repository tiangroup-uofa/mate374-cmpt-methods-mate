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
    # Conditioning: when `Ax=b` is sensitive

    A linear solve can have a tiny residual and still produce a sensitive
    physical inference. Imagine two material signatures that are almost
    indistinguishable in two measurements:

    $$
    A\mathbf{x}=\mathbf{b},\qquad
    A=\begin{bmatrix}1&1\\1&1+\delta\end{bmatrix}.
    $$

    The unknown vector contains two contributions. As $\delta\to0$, the
    columns of $A$ become nearly dependent. A tiny measurement perturbation
    can then cause a large change in $\mathbf{x}$.

    This is **conditioning**, not simply a programming bug. We will compare
    the residual of the computed system with the error relative to the clean
    material model.
    """)
    return


@app.cell
def _(mo):
    separation = mo.ui.dropdown(
        options={
            "δ = 10⁻¹": 1.0e-1,
            "δ = 10⁻²": 1.0e-2,
            "δ = 10⁻³": 1.0e-3,
            "δ = 10⁻⁴": 1.0e-4,
        },
        value="δ = 10⁻³",
        label="Separation between material signatures",
    )
    measurement_noise = mo.ui.dropdown(
        options={
            "no perturbation": 0.0,
            "10⁻⁶": 1.0e-6,
            "10⁻⁴": 1.0e-4,
            "10⁻³": 1.0e-3,
        },
        value="10⁻⁴",
        label="Perturbation in the second measurement",
    )
    mo.vstack([separation, measurement_noise], align="start", gap=1)
    return measurement_noise, separation


@app.cell
def _(np):
    def condition_experiment(separation, measurement_noise):
        true_coefficients = np.array([0.6, 0.4])
        matrix = np.array(
            [[1.0, 1.0], [1.0, 1.0 + separation]], dtype=float
        )
        clean_observations = matrix @ true_coefficients
        noisy_observations = clean_observations + np.array(
            [0.0, measurement_noise]
        )
        recovered_clean = np.linalg.solve(matrix, clean_observations)
        recovered_noisy = np.linalg.solve(matrix, noisy_observations)
        row_scales = np.linalg.norm(matrix, axis=1)
        scaled_matrix = matrix / row_scales[:, None]
        return (
            clean_observations,
            matrix,
            noisy_observations,
            recovered_clean,
            recovered_noisy,
            scaled_matrix,
            true_coefficients,
        )

    return (condition_experiment,)


@app.cell
def _(condition_experiment, measurement_noise, mo, np, separation):
    (
        clean_observations,
        matrix,
        noisy_observations,
        recovered_clean,
        recovered_noisy,
        scaled_matrix,
        true_coefficients,
    ) = condition_experiment(separation.value, measurement_noise.value)

    clean_condition = np.linalg.cond(matrix, 2)
    scaled_condition = np.linalg.cond(scaled_matrix, 2)
    noisy_residual = matrix @ recovered_noisy - noisy_observations
    clean_model_mismatch = matrix @ recovered_noisy - clean_observations
    coefficient_change = recovered_noisy - true_coefficients
    result_kind = "warn" if clean_condition > 1.0e3 else "info"
    mo.callout(
        mo.md(
            f"""
            ## Sensitivity result

            - matrix 2-norm condition number: **{clean_condition:.3e}**
            - row-scaled condition number: **{scaled_condition:.3e}**
            - clean-data solution: **{np.array2string(recovered_clean, precision=4)}**
            - perturbed-data solution: **{np.array2string(recovered_noisy, precision=4)}**
            - change from the true coefficients: **{np.array2string(coefficient_change, precision=4)}**
            - residual for the perturbed system: **{np.linalg.norm(noisy_residual, ord=np.inf):.3e}**
            - mismatch against the clean model: **{np.linalg.norm(clean_model_mismatch, ord=np.inf):.3e}**

            The solver can satisfy the perturbed equations almost exactly while
            the inferred coefficients move substantially. A small residual alone
            is not a sensitivity or uncertainty analysis.
            """
        ),
        kind=result_kind,
    )
    return clean_observations, matrix, noisy_observations


@app.cell
def _(clean_observations, matrix, mo, noisy_observations, np):
    mo.md(
        f"""
        ### Inspect the data story

        ```text
        A =
        {np.array2string(matrix, precision=6)}

        clean observations  = {np.array2string(clean_observations, precision=6)}
        perturbed observations = {np.array2string(noisy_observations, precision=6)}
        ```

        Row scaling can make units and magnitudes more comparable, but it cannot
        create information that the two measurements do not contain. Nearly
        dependent columns remain a physical identifiability problem.
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### Trust checklist

            For a reported linear solve, show more than `x`:

            - the units and scaling of `A` and `b`;
            - a residual in the equations actually solved;
            - sensitivity to plausible data perturbations;
            - a condition or identifiability diagnostic;
            - whether the inferred coefficients remain physically admissible.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()

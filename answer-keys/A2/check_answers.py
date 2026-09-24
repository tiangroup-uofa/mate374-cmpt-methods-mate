"""Run the completed A2 notebooks and regenerate answer tables, code, and figures.

Run from the repository root: uv run --locked python answer-keys/A2/check_answers.py
No CSV or student notebook is modified. All saved figures use 300 dpi.
"""
from pathlib import Path
import ast
import json
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"


def snippet(notebook, function, filename, prefix=""):
    tree = ast.parse(notebook.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    (GENERATED / filename).write_text("```python\n" + prefix + ast.unparse(node) + "\n```\n")


def table(filename, headings, rows):
    text = "| " + " | ".join(headings) + " |\n"
    text += "| " + " | ".join("---" for _ in headings) + " |\n"
    text += "".join("| " + " | ".join(map(str, row)) + " |\n" for row in rows)
    (GENERATED / filename).write_text(text)


def save(fig, filename):
    fig.savefig(HERE / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    GENERATED.mkdir(exist_ok=True)
    paths = {q: HERE / "notebooks" / f"a2_q{q}_completed.edit.py" for q in (2, 3, 4)}
    definitions = {}
    for q, path in paths.items():
        _, definitions[q] = runpy.run_path(str(path))["app"].run()
        plt.close("all")
    q2, q3, q4 = (definitions[q] for q in (2, 3, 4))
    sequence = q2["sequence"]
    assert len(sequence) - 1 == 12
    assert abs(sequence[-1] - sequence[-2]) < 1e-4
    assert abs(sequence[-2] - sequence[-3]) >= 1e-4
    assert np.isclose(q2["g"](0.5), 0.5)
    assert q3["objective_ready"] and q3["fit"].success
    assert q4["residual_ok"] and q4["parity_values"] is not None
    assert len(q4["clusters"]) == 100 and len(q4["ar20_clusters"]) == 5
    assert {c["n_atoms"] for c in q4["clusters"]} == {3, 4, 5, 6, 7}
    assert {c["n_atoms"] for c in q4["ar20_clusters"]} == {20}

    for q, functions in {
        2: ["g", "fixed_point"],
        3: ["estimate_by_interpolation", "squared_residual"],
        4: ["cluster_residual", "fit_LJ", "parity_energies"],
    }.items():
        for function in functions:
            snippet(paths[q], function, f"{function}.qmd", prefix="H = 2.5\n\n" if function == "g" else "")

    table("q2-iterations.qmd", ["Update $n$", "$x_n$", "$|x_n-x_{n-1}|$"],
          [[n, f"{x:.12f}", "—" if n == 0 else f"{abs(x-sequence[n-1]):.6g}"]
           for n, x in enumerate(sequence)])
    beta = float(sequence[-1])
    q2_summary = {"beta": beta, "updates": len(sequence)-1,
                  "last_step": abs(sequence[-1]-sequence[-2]),
                  "residual": float(q2["F"](beta, 2.5))}
    (GENERATED / "q2-summary.qmd").write_text(
        f"The step tolerance is first met after **{len(sequence)-1} updates**, giving "
        f"$x_{{12}}={beta:.12f}$, or **$x_\\beta={beta:.5g}$** to five significant digits. "
        f"The final step is ${q2_summary['last_step']:.6g}$ and the residual at the "
        f"unrounded iterate is $F(x_{{12}})={q2_summary['residual']:+.8f}$.\n")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    grid = np.linspace(0, 1, 600)
    for H, ax in zip([1.5, 2.5], axes):
        _, comparison = runpy.run_path(str(paths[2]))["app"].run(defs={"H": H})
        mapping = comparison["g"]
        ax.plot(grid, mapping(grid), label="g(x)")
        ax.plot(grid, grid, "--", color="0.4", label="y = x")
        ax.set(xlabel="x", ylabel="y", title=f"H = {H}", xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
        ax.legend()
    save(fig, "fixed-point.png")

    interpolation = q3["interpolation_rows"]
    table("q3-interpolation.qmd", ["Method", "$\\sigma$ from root (Å)", "$\\sigma$ from minimum (Å)", "$\\varepsilon$ from minimum (eV)"],
          [[row["Method"], f"{row['sigma_root (Å)']:.7f}", f"{row['sigma_min (Å)']:.7f}",
            f"{row['epsilon_min (eV)']:.9f}"] for row in interpolation])
    table("q3-locations.qmd", ["Method", "$r_0$ (Å)", "$r_m$ (Å)", "$V_m$ (eV)"],
          [[row["Method"], f"{row['r_zero (Å)']:.7f}", f"{row['r_min (Å)']:.7f}",
            f"{row['V_min (eV)']:.9f}"] for row in interpolation])
    fit3 = q3["fit"]
    sigma3, epsilon3 = map(float, fit3.x)
    (GENERATED / "q3-fit.qmd").write_text(
        f"The fit gives **$\\sigma={sigma3:.8f}$ Å** and "
        f"**$\\varepsilon={epsilon3:.9f}$ eV**, with total squared residual "
        f"${float(fit3.fun):.7g}$ eV$^2$. The optimizer reports success.\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    r, V = q3["r_data"], q3["V_data"]
    grid = np.linspace(r[0], r[-1], 800)
    methods = [q3["CubicSpline"](r, V, bc_type="natural"),
               q3["PchipInterpolator"](r, V), q3["interp1d"](r, V)]
    for ax in axes:
        ax.scatter(r, V, s=20, color="black", label="Data", zorder=4)
        for name, method in zip(["Natural cubic", "PCHIP", "Linear"], methods):
            ax.plot(grid, method(grid), label=name, lw=1.3)
        ax.plot(grid, q3["lj"](grid, epsilon3, sigma3), "--", color="black", label="LJ fit")
        ax.set(xlabel="Distance r (Å)", ylabel="Energy V (eV)")
        ax.grid(alpha=.2)
    axes[0].axhline(0, color="0.5", lw=.6)
    axes[0].legend(fontsize=7)
    axes[1].set(xlim=(3.55, 4.3), ylim=(-.011, -.008), title="Near the energy minimum")
    save(fig, "lj-fit.png")

    fit_rows = q4["fit_rows"]
    table("q4-fits.qmd", ["Clusters $N$", "$\\sigma$ (Å)", "$\\varepsilon$ (eV)", "MAE (meV/atom)"],
          [[row["N"], f"{row['sigma (Å)']:.8f}", f"{row['epsilon (eV)']:.9f}",
            f"{row['MAE (meV/atom)']:.6f}"] for row in fit_rows])
    sigma4, epsilon4 = q4["best_sigma"], q4["best_epsilon"]
    total_mae, atom_mae = q4["energy_errors"](q4["ar20_clusters"], sigma4, epsilon4)
    assert np.isclose(atom_mae, total_mae * 1000 / 20)
    qm20, lj20 = q4["parity_values"][1]
    table("q4-extrapolation.qmd", ["Ar₂₀ cluster", "$E^{\\mathrm{QM}}$ (eV)", "$E^{\\mathrm{LJ}}$ (eV)", "Absolute error (meV/atom)"],
          [[i+1, f"{qm:.8f}", f"{lj:.8f}", f"{abs(lj-qm)*50:.5f}"]
           for i, (qm, lj) in enumerate(zip(qm20, lj20))])
    (GENERATED / "q4-summary.qmd").write_text(
        f"Using the **100-cluster fit**, the five Ar₂₀ clusters have a mean absolute "
        f"total-energy difference of **{total_mae:.8f} eV/cluster** and a mean absolute "
        f"per-atom difference of **{atom_mae:.6f} meV/atom**. The corresponding fitting-set "
        f"MAE is **{fit_rows[-1]['MAE (meV/atom)']:.6f} meV/atom**.\n")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), layout="constrained")
    for ax in axes:
        for (qm, lj), label, marker in zip(q4["parity_values"], ["100 Ar₃–Ar₇", "5 Ar₂₀"], ["o", "^"]):
            ax.scatter(qm, lj, label=label, marker=marker, s=23, alpha=.8)
        ax.set(xlabel="QM total energy (eV)", ylabel="LJ total energy (eV)")
        ax.grid(alpha=.2)
        ax.set_aspect("equal", adjustable="box")
    values = np.concatenate([a for pair in q4["parity_values"] for a in pair])
    limits = (float(values.min())-.03, float(values.max())+.03)
    for ax, lim in zip(axes, [limits, (-.15, .08)]):
        ax.plot(lim, lim, "--", color="0.4")
        ax.set(xlim=lim, ylim=lim)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Zoom near the smaller-cluster energies")
    save(fig, "cluster-parity.png")
    results = {"q2": q2_summary,
               "q3": {"interpolation": interpolation, "sigma": sigma3, "epsilon": epsilon3,
                       "residual": float(fit3.fun)},
               "q4": {"fits": fit_rows, "test_total_mae_eV": total_mae,
                       "test_mae_meV_per_atom": atom_mae,
                       "test_qm_eV": qm20.tolist(), "test_lj_eV": lj20.tolist()}}
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

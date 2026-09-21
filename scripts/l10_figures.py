"""Check the L10 scaffold, copyable solutions, and multi-gas calculations.

    uv run --locked python scripts/l10_figures.py [--check-only]
"""
import argparse
import ast
import json
from pathlib import Path
import re
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    app = runpy.run_path(str(ROOT / "activities" / f"{name}.edit.py"))["app"]
    _, definitions = app.run()
    return definitions


def check_solution(d):
    solve = d["solve_pressure"]
    R = d["R"]
    np.testing.assert_allclose(d["trial_P"],
                               [33.7477377234, 47.2821043438, 63.7220255117], atol=1e-7)
    assert d["checks_passed"]
    assert abs(d["interpolated_P"] - d["direct_P"]) < 0.01
    assert abs(d["fitted_P"] - d["direct_P"]) < 0.2
    assert not np.any(np.isclose(d["sample_T"], d["check_T"]))
    assert np.isnan(d["boundary"](300))

    data = json.loads((ROOT / "data/L10-gases.json").read_text())
    for name, gas in data.items():
        embedded = d["gases"][name]
        assert (embedded["a"], embedded["b"]) == (gas["a"], gas["b"])
        if gas["reference"] is not None:
            for key in ("A", "B", "C", "T_min", "T_max"):
                assert embedded["reference"][key] == gas["reference"][key]
        else:
            assert embedded["reference"] is None
        a, b = gas["a"], gas["b"]
        Tc = 8*a/(27*R*b)
        pressures = []
        # Cover endpoints and the trial/interpolation ranges for every substance.
        for fraction in [0.820001, 0.84, 0.85, 0.90, 0.96, 0.984999]:
            T = fraction*Tc
            low, high = d["find_pressure_bracket"](T, a, b)
            delta = lambda P: d["free_energy_difference_P"](P, T, a, b)
            assert delta(low)*delta(high) < 0
            P = solve(T, a, b)
            pressures.append(P)
            assert abs(delta(P)) < 1e-4
            # Independent volume calculation: solve the EOS cubic, then
            # check Maxwell's equal-area condition by quadrature.
            roots = np.roots([P, -(P*b+R*T), a, -a*b])
            assert np.max(np.abs(roots.imag)) < 1e-9
            volumes = np.sort(roots.real)
            assert volumes[0] > b
            area, _ = quad(lambda v: R*T/(v-b)-a/v**2-P,
                           volumes[0], volumes[-1], epsabs=1e-10)
            assert abs(area) < 1e-6, (name, fraction, area)
            minima = d["find_phase_volumes"](
                lambda v: d["free_energy"](v, T, P, a, b), T, P, a, b)
            np.testing.assert_allclose(minima, volumes[[0, 2]], rtol=2e-6)
            np.testing.assert_allclose(d["pressure_eos"](np.array(minima), T, a, b),
                                       P, atol=1e-3, rtol=0)
        assert np.all(np.diff(pressures) > 0)
        for bad_T in [0, -1, np.nan, 0.8*Tc, 0.99*Tc, Tc, 1.1*Tc]:
            try:
                solve(bad_T, a, b)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted unsupported T={bad_T}")

    for row in d["gas_comparison"]:
        assert abs(row["gap"]) < 1e-4
        if row["reference"] is not None:
            ref = data[row["gas"]]["reference"]
            assert ref["T_min"] <= row["T"] <= ref["T_max"]
            np.testing.assert_allclose(row["reference"],
                10**(ref["A"]-ref["B"]/(row["T"]+ref["C"])))
            print(f"{row['gas']}: vdW {row['P']:.4f} bar; NIST {row['reference']:.4f} bar")


def check_starter_and_copyable(d):
    starter = load("l10_phase_diagram")
    assert starter["trial_P"] == [None, None, None]
    # Only the introduction and the two student tasks may differ.
    def cells(name):
        tree = ast.parse((ROOT / "activities" / f"{name}.edit.py").read_text())
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    left, right = cells("l10_phase_diagram"), cells("l10_phase_diagram_solution")
    assert left.keys() == right.keys()
    assert {key for key in left if left[key] != right[key]} == {
        "introduction", "pressure_solver", "interpolation"}

    page = (ROOT / "units/02/L10/index.qmd").read_text()
    blocks = re.findall(r"```\{\.python[^\n]*\}\n(.*?)\n```", page, re.S)
    scope = dict(d)
    for block in blocks:
        exec(compile(block, "L10-copyable-code", "exec"), scope)
    np.testing.assert_allclose(scope["solve_pressure"](280, 3.592, 0.04267), 55.12905257)
    np.testing.assert_allclose(scope["boundary"](275), d["interpolated_P"])

    # Exercise the actual starter cells after pasting the completed code.
    app = runpy.run_path(str(ROOT / "activities/l10_phase_diagram.edit.py"))["app"]
    _, completed = app.run(defs={"solve_pressure": scope["solve_pressure"],
                                  "boundary": scope["boundary"]})
    assert completed["checks_passed"]
    np.testing.assert_allclose(completed["trial_P"], d["trial_P"])
    print(f"Starter stops cleanly; completed replacements and {len(blocks)} page blocks pass.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    d = load("l10_phase_diagram_solution")
    check_solution(d)
    check_starter_and_copyable(d)
    if not args.check_only:
        path = ROOT / "assets/L10-phase-boundary.png"
        d["phase_figure"].savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path.relative_to(ROOT)}")
    plt.close("all")
    print("L10 numerical checks passed.")


if __name__ == "__main__":
    main()

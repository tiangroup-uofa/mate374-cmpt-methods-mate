"""Check the L10 scaffold, copyable solutions, and CO₂ fit and coexistence calculations.

    uv run --locked python scripts/l10_figures.py [--check-only]
"""
import argparse
import ast
import csv
from pathlib import Path
import re
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    app = runpy.run_path(str(ROOT / "activities" / f"{name}.edit.py"))["app"]
    _, definitions = app.run()
    return dict(definitions)


def check_data(d):
    # The calculation inputs must match the downloadable CSV.
    rows = list(csv.DictReader((ROOT / "data/L10-co2-pvt.csv").open()))
    np.testing.assert_allclose(d["T_data"], [float(r["T_K"]) for r in rows])
    np.testing.assert_allclose(d["P_data"], [float(r["P_bar"]) for r in rows], rtol=1e-12)
    np.testing.assert_allclose(d["v_data"], [float(r["V_L_mol"]) for r in rows], rtol=1e-12)
    np.testing.assert_array_equal(d["fit_set"], d["rho_data"] < 10)
    assert d["fit_set"].sum() == 24
    assert np.all(d["rho_data"][d["fit_set"]] < 10)


def check_solution(d):
    solve = d["solve_pressure"]
    R = d["R"]
    np.testing.assert_allclose([d["a_fit"], d["b_fit"]], [3.58867, 0.042212], rtol=2e-5)
    np.testing.assert_allclose(d["trial_P"], [32.9521642, 46.2704879, 62.4754738], atol=1e-6)
    assert d["checks_passed"]
    np.testing.assert_allclose(d["boundary"](275), np.interp(275, d["T_grid"], d["P_grid"]))
    assert abs(d["interpolated_P"] - d["direct_P"]) < 0.1
    np.testing.assert_allclose(d["fit_rmse"], 1.79932274, atol=1e-6)
    assert not d["data_figure"].axes[0].get_title()
    assert not np.any(np.isclose(d["T_grid"], d["check_T"]))
    assert np.isnan(d["boundary"](300))
    # Span–Wagner ancillary vapour pressure at 280 K is 41.6 bar.
    np.testing.assert_allclose(d["measured_saturation_pressure"](280.0), 41.607, atol=0.01)

    for a, b in [(d["a_fit"], d["b_fit"]), (d["a_book"], d["b_book"])]:
        Tc = 8*a/(27*R*b)
        pressures = []
        # Cover the scaffold limits and the lecture's 250–295 K range.
        for T in [0.820001*Tc, 250.0, 270.0, 275.0, 290.0, 295.0, 0.984999*Tc]:
            low, high = d["find_pressure_bracket"](T, a, b)
            delta = lambda P: d["delta_G"](P, T, a, b)
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
            assert abs(area) < 1e-6, (a, b, T, area)
            minima = d["phase_volumes"](T, P, a, b)
            np.testing.assert_allclose(minima, volumes[[0, 2]], rtol=2e-6)
        assert np.all(np.diff(pressures[1:-1]) > 0)
        for bad_T in [0, -1, np.nan, 0.8*Tc, 0.99*Tc, Tc, 1.1*Tc]:
            try:
                solve(bad_T, a, b)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted unsupported T={bad_T}")

    print(f"Fitting states: {d['fit_set'].sum()}")
    print(f"275 K: direct {d['direct_P']:.4f}, linear {d['interpolated_P']:.4f} bar")
    residual = d["pressure_eos"](d["v_data"], d["T_data"], d["a_fit"], d["b_fit"]) - d["P_data"]
    for label, mask in [("rho < 10", d["fit_set"]), ("rho >= 10", ~d["fit_set"])]:
        print(f"{label}: pressure RMSE {np.sqrt(np.mean(residual[mask]**2)):.2f} bar")
    for T, fitted, measured in zip(d["answer_T"], d["answer_fit"], d["answer_measured"]):
        print(f"{T:g} K: fitted vdW {fitted:.2f} bar; measured {measured:.2f} bar")


def check_starter_and_copyable(d):
    starter = load("l10_phase_diagram")
    assert starter["a_fit"] is None and "trial_P" not in starter
    check_data(starter)
    # Only the introduction and the four live-coded cells may differ.
    def cells(name):
        tree = ast.parse((ROOT / "activities" / f"{name}.edit.py").read_text())
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    left, right = cells("l10_phase_diagram"), cells("l10_phase_diagram_solution")
    assert left.keys() == right.keys()
    assert {key for key in left if left[key] != right[key]} == {
        "introduction", "live_fit", "live_minimize", "live_root", "live_boundary"}

    page = (ROOT / "units/02/L10/index.qmd").read_text()
    blocks = re.findall(r"```\{\.python[^\n]*\}\n(.*?)\n```", page, re.S)
    # Check the page examples numerically: their helper structure and selected
    # methods may differ from the reference notebook.
    scope = dict(d)
    for block in blocks:
        exec(compile(block, "L10-copyable-code", "exec"), scope)
    np.testing.assert_allclose([scope["a_fit"], scope["b_fit"]], [d["a_fit"], d["b_fit"]])
    np.testing.assert_allclose(scope["solve_pressure"](280, d["a_fit"], d["b_fit"]),
                               d["answer_fit"][3])
    np.testing.assert_allclose(scope["boundary"](275),
                               np.interp(275, d["T_grid"], d["P_grid"]))
    for method in ("linear", "cubic", "pchip"):
        boundary = scope["interpolate_boundary"](d["T_grid"], d["P_grid"], method)
        np.testing.assert_allclose(boundary(d["T_grid"]), d["P_grid"])
        assert np.isnan(boundary(300))
        assert abs(boundary(275) - d["direct_P"]) < 0.1

    # Exercise the actual starter cells after pasting the completed code.
    stages = [
        ["vdw_pressure", "a_fit", "b_fit"],
        ["find_state_volume", "free_energy", "phase_volumes"],
        ["delta_G", "solve_pressure"],
        ["T_grid", "P_grid", "boundary"],
    ]
    replacements = {}
    for stage in stages:
        replacements.update({name: scope[name] for name in stage})
        app = runpy.run_path(str(ROOT / "activities/l10_phase_diagram.edit.py"))["app"]
        _, completed = app.run(defs=replacements)
        if "solve_pressure" in replacements:
            assert completed["checks_passed"]
            np.testing.assert_allclose(completed["trial_P"], d["trial_P"])
        plt.close("all")

    # The tutor and filled student versions must respond to all three choices.
    for method in ("linear", "cubic", "pchip"):
        boundary = d["interpolate_boundary"](d["T_grid"], d["P_grid"], method)
        for name in ("l10_phase_diagram", "l10_phase_diagram_solution"):
            app = runpy.run_path(str(ROOT / "activities" / f"{name}.edit.py"))["app"]
            _, switched = app.run(defs={**replacements, "boundary": boundary})
            np.testing.assert_allclose(switched["interpolated_P"], boundary(275))
            assert switched["checks_passed"]
            plt.close("all")
    print(f"Starter stops cleanly; completed replacements and {len(blocks)} page blocks pass.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    d = load("l10_phase_diagram_solution")
    check_data(d)
    check_solution(d)
    check_starter_and_copyable(d)
    if not args.check_only:
        for key, name in [("data_figure", "L10-co2-measurements.png"),
                          ("phase_figure", "L10-phase-boundary.png")]:
            path = ROOT / "assets" / name
            d[key].savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved {path.relative_to(ROOT)}")
    plt.close("all")
    print("L10 numerical checks passed.")


if __name__ == "__main__":
    main()

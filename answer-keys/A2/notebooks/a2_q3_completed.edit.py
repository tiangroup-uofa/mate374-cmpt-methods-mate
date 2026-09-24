# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy", "matplotlib>=3.11"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
    from scipy.optimize import root_scalar, minimize_scalar, minimize
    return CubicSpline, PchipInterpolator, interp1d, minimize, minimize_scalar, mo, np, plt, root_scalar


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # A2 Q3 · Completed reference notebook

    How well can eight sampled energies determine $\varepsilon$ and $\sigma$?
    Compare estimates from an interpolated curve with a least-squares fit of
    the LJ function from Assignment 1. The data and LJ function are supplied.
    Both functions are completed. Compare all three interpolation methods
    with the least-squares fit.

    Reference values: $\varepsilon_{\mathrm{ref}}=0.0103$ eV and
    $\sigma_{\mathrm{ref}}=3.40$ Å.
    """)
    return


@app.cell(hide_code=True)
def supplied_model(np):
    r_data = np.array([3.20, 3.35, 3.50, 3.65, 4.00, 4.40, 5.00, 6.00])
    V_data = np.array([0.026084565, 0.004126033, -0.005477196, -0.009421694,
                  -0.009608200, -0.006943821, -0.003610619, -0.001348986])

    def lj(r, epsilon, sigma):
        return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

    return V_data, lj, r_data


@app.cell(hide_code=True)
def interpolation_instructions(mo):
    mo.md(r"""
    ## Q3.1 · Estimates from interpolation

    Choose an interpolation method and inspect the curve through the data.
    The supplied code creates `f_interp(r)`, which gives an energy in eV.
    The default is a natural cubic spline. You can also compare
    a shape-preserving cubic (PCHIP) and piecewise linear interpolation.
    """)
    return


@app.cell(hide_code=True)
def supplied_interpolation(CubicSpline, PchipInterpolator, V_data, interp1d, interpolation_method, r_data):
    if interpolation_method.value == "Natural cubic spline":
        f_interp = CubicSpline(r_data, V_data, bc_type="natural", extrapolate=False)
    elif interpolation_method.value == "Shape-preserving cubic (PCHIP)":
        f_interp = PchipInterpolator(r_data, V_data, extrapolate=False)
    else:
        f_interp = interp1d(r_data, V_data, kind="linear", bounds_error=True)
    return (f_interp,)


@app.cell(hide_code=True)
def method_control(mo):
    interpolation_method = mo.ui.dropdown(
        options=["Natural cubic spline", "Shape-preserving cubic (PCHIP)", "Piecewise linear"],
        value="Natural cubic spline", label="Interpolation method",
    )
    interpolation_method
    return (interpolation_method,)


@app.cell(hide_code=True)
def interpolation_plot(V_data, f_interp, interpolation_method, np, plt, r_data):
    _grid = np.linspace(r_data[0], r_data[-1], 600)
    _fig, _ax = plt.subplots(figsize=(7, 4), layout="constrained")
    _ax.scatter(r_data, V_data, color="black", label="Sampled energies", zorder=3)
    _ax.plot(_grid, f_interp(_grid), label=interpolation_method.value)
    _ax.axhline(0, color="0.5", lw=0.8)
    _ax.set(xlabel="Distance r (Å)", ylabel="Energy V (eV)")
    _ax.legend()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def estimate_instructions(mo):
    mo.md(r"""
    ### Q3.1 · Complete `estimate_by_interpolation`

    `estimate_by_interpolation(method)` is a scaffold. The first part selects
    the interpolation method and creates `f_interp(r)`. **Change only the code
    below the marked line.** Use `root_scalar` to find the zero crossing in
    $[3.35,3.50]$ Å and `minimize_scalar` to find the minimum in $[3.50,4.40]$ Å.
    **Both functions are already imported.** Read the SciPy documentation for
    [root_scalar](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html)
    and [minimize_scalar](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html)
    to find out how to use them. Return three scalar numbers, in this order:

    - `r_zero`: the zero-crossing distance in Å;
    - `r_min`: the distance at the minimum in Å;
    - `V_min`: the interpolated energy at that minimum in eV.

    The starting zeros leave the parameter estimates locked. Once your returned
    values pass the checks, the supplied conversion uses the A1 relations
    $\sigma=r_0$, $\sigma=r_m/2^{1/6}$, and $\varepsilon=-V(r_m)$.

    Copy your working code for `estimate_by_interpolation` into your answer
    report, including the results from all three methods.
    """)
    return


@app.cell(hide_code=False)
def student_estimate(CubicSpline, PchipInterpolator, V_data, interp1d, minimize_scalar, r_data, root_scalar):
    def estimate_by_interpolation(method):
        # Provided: select the method and create the energy function f_interp(r).
        if method == "Natural cubic spline":
            f_interp = CubicSpline(r_data, V_data, bc_type="natural", extrapolate=False)
        elif method == "Shape-preserving cubic (PCHIP)":
            f_interp = PchipInterpolator(r_data, V_data, extrapolate=False)
        elif method == "Piecewise linear":
            f_interp = interp1d(r_data, V_data, kind="linear", bounds_error=True)
        else:
            raise ValueError("Unknown interpolation method")

        # ===== CHANGE ONLY THE CODE BELOW THIS LINE =====
        # root_scalar and minimize_scalar are ALREADY IMPORTED.
        # Read their SciPy documentation (linked above) to learn how to use them.
        # Use these functions with f_interp.
        # Evaluate f_interp at the minimum to obtain V_min.
        zero = root_scalar(f_interp, bracket=[3.35, 3.50], method="brentq")
        minimum = minimize_scalar(
            f_interp, bounds=[3.50, 4.40], method="bounded",
            options={"xatol": 1e-12},
        )
        if not zero.converged or not minimum.success:
            raise RuntimeError("Root finding or minimization failed")
        r_zero = float(zero.root)
        r_min = float(minimum.x)
        V_min = float(f_interp(r_min))
        return r_zero, r_min, V_min

    return (estimate_by_interpolation,)


@app.cell(hide_code=True)
def check_estimate(estimate_by_interpolation, f_interp, interpolation_method, mo, np):
    estimate_values = None
    try:
        _raw = np.asarray(estimate_by_interpolation(interpolation_method.value))
        if _raw.shape != (3,) or not np.all(np.isfinite(_raw.astype(float))):
            raise ValueError("Return three finite scalar numbers: r_zero, r_min, V_min.")
        _zero, _minimum, _energy = _raw.astype(float)
        if not (3.35 <= _zero <= 3.50 and 3.50 <= _minimum <= 4.40):
            raise ValueError("The returned distances must lie in the specified search intervals.")
        if abs(float(f_interp(_zero))) > 1e-8:
            raise ValueError("Check the zero crossing: the interpolated energy there should be close to zero.")
        if abs(_energy - float(f_interp(_minimum))) > 1e-8:
            raise ValueError("V_min should be the interpolated energy at your r_min.")
        _grid = np.linspace(3.50, 4.40, 10001)
        if _energy >= 0 or _energy > float(np.min(f_interp(_grid))) + 1e-8:
            raise ValueError("Check the minimization: lower energies occur within the search interval.")
        estimate_values = (_zero, _minimum, _energy)
        _message = mo.callout(mo.md("Your crossing and minimum pass the numerical checks."), kind="success")
    except Exception as _error:
        _message = mo.callout(mo.md(f"**Parameter estimates locked.** {_error}"), kind="warn")
    _message
    return (estimate_values,)


@app.cell(hide_code=True)
def supplied_conversion():
    def convert_to_parameters(r_zero, r_min, V_min):
        sigma_root = r_zero
        sigma_minimum = r_min / 2**(1 / 6)
        epsilon_minimum = -V_min
        return sigma_root, sigma_minimum, epsilon_minimum

    return (convert_to_parameters,)


@app.cell(hide_code=True)
def converted_estimates(convert_to_parameters, estimate_values, interpolation_method, mo):
    mo.stop(estimate_values is None)
    _sigma_root, _sigma_minimum, _epsilon_minimum = convert_to_parameters(*estimate_values)
    mo.md(
        "### Q3.1 · Parameter estimates\n\n"
        f"**Method: {interpolation_method.value}**\n\n"
        "| Estimate | Value | Reference |\n|---|---:|---:|\n"
        f"| σ from zero crossing (Å) | {_sigma_root:.6g} | 3.40 |\n"
        f"| σ from minimum location (Å) | {_sigma_minimum:.6g} | 3.40 |\n"
        f"| ε from minimum energy (eV) | {_epsilon_minimum:.6g} | 0.0103 |\n\n"
        "Record these results for each of the three methods. Which estimates are closest to the reference values?"
    )
    return


@app.cell(hide_code=True)
def fitting_instructions(mo):
    mo.md(r"""
    ## Q3.2 · Least-squares fitting

    Complete the scaffold **`squared_residual(sigma, epsilon)`**.
    The arrays `r_data` (Å) and `V_data` (eV) are already defined outside
    the function. Use `lj(r, epsilon, sigma)` to calculate the model energies
    at these distances and add up the squared differences. Return **one scalar**:

    $$E(\varepsilon,\sigma)=\sum_{i=1}^{8}[V_i-V(r_i;\varepsilon,\sigma)]^2.$$

    The fitting call and comparison plot unlock when your function passes the
    checks. We supply `minimize`, positive parameter bounds, and the starting
    pair $\varepsilon=0.01$ eV, $\sigma=3.3$ Å. The fit uses all eight data points.

    Copy your working code for `squared_residual` into your answer report,
    including the fitted parameters. Compare them with the reference values
    and your interpolation estimates.
    """)
    return


@app.cell(hide_code=False)
def student_objective(V_data, lj, np, r_data):
    def squared_residual(sigma, epsilon):
        # Use the global arrays r_data and V_data.
        # Use lj(r_i, epsilon, sigma) to get the model energy at each distance.
        total = 0.0
        for r_i, V_i in zip(r_data, V_data):
            V_model = lj(r_i, epsilon, sigma)
            total += (V_i - V_model)**2
        return total

    return (squared_residual,)


@app.cell(hide_code=True)
def check_objective(V_data, lj, mo, np, r_data, squared_residual):
    objective_ready = False
    try:
        for _sigma, _epsilon in [(3.25, 0.009), (3.5, 0.012), (3.35, 0.011)]:
            _value = squared_residual(_sigma, _epsilon)
            if _value is Ellipsis or np.asarray(_value).shape != ():
                raise ValueError("Return one scalar sum, rather than an array of residuals or `...`.")
            _value = float(_value)
            _difference = V_data - lj(r_data, _epsilon, _sigma)
            _expected = float(np.dot(_difference, _difference))
            if not np.isfinite(_value) or not np.isclose(_value, _expected, rtol=1e-7, atol=1e-12):
                raise ValueError("Check the sum of squared differences using r_data, V_data, sigma, and epsilon.")
        objective_ready = True
        _message = mo.callout(mo.md("Your scalar objective passes the checks."), kind="success")
    except Exception as _error:
        _message = mo.callout(mo.md(f"**Least-squares fit locked.** {_error}"), kind="warn")
    _message
    return (objective_ready,)


@app.cell(hide_code=True)
def supplied_fit(minimize, mo, objective_ready, squared_residual):
    mo.stop(not objective_ready)
    fit = minimize(
        lambda parameters: squared_residual(parameters[0], parameters[1]),
        x0=[3.3, 0.01], method="Nelder-Mead",
        bounds=[(1e-6, None), (1e-6, None)],
        options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 3000},
    )
    mo.md(
        "### Q3.2 · Fitted parameters\n\n"
        f"**Converged:** {fit.success} ({fit.message})\n\n"
        "| Fitted parameter | Value | Reference |\n|---|---:|---:|\n"
        f"| ε (eV) | {fit.x[1]:.6g} | 0.0103 |\n"
        f"| σ (Å) | {fit.x[0]:.6g} | 3.40 |\n\n"
        f"Total squared residual: {fit.fun:.6g} eV²."
    )
    return (fit,)


@app.cell(hide_code=True)
def fitted_plot(V_data, f_interp, fit, interpolation_method, lj, mo, np, plt, r_data):
    mo.stop(not fit.success, mo.md("Check the optimizer status before interpreting the fitted parameters."))
    _grid = np.linspace(r_data[0], r_data[-1], 600)
    _fig, _ax = plt.subplots(figsize=(7, 4), layout="constrained")
    _ax.scatter(r_data, V_data, color="black", label="Sampled energies", zorder=3)
    _ax.plot(_grid, f_interp(_grid), label=interpolation_method.value)
    _ax.plot(_grid, lj(_grid, fit.x[1], fit.x[0]), "--", label="Least-squares LJ fit")
    _ax.axhline(0, color="0.5", lw=0.8)
    _ax.set(xlabel="Distance r (Å)", ylabel="Energy V (eV)")
    _ax.legend()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def all_method_results(convert_to_parameters, estimate_by_interpolation, mo):
    interpolation_rows = []
    for method in ["Natural cubic spline", "Shape-preserving cubic (PCHIP)", "Piecewise linear"]:
        r_zero, r_min, V_min = estimate_by_interpolation(method)
        sigma_root, sigma_min, epsilon_min = convert_to_parameters(r_zero, r_min, V_min)
        interpolation_rows.append({"Method": method, "r_zero (Å)": r_zero,
                                   "r_min (Å)": r_min, "V_min (eV)": V_min,
                                   "sigma_root (Å)": sigma_root, "sigma_min (Å)": sigma_min,
                                   "epsilon_min (eV)": epsilon_min})
    mo.vstack([mo.md("## Q3.1 · All three methods"),
               mo.ui.table(interpolation_rows, selection=None)])
    return (interpolation_rows,)


if __name__ == "__main__":
    app.run()

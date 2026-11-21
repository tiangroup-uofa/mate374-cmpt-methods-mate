# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.16", "numpy>=2.0", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from io import StringIO

    return StringIO, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which time window gives a useful diffusion estimate?

    **Predict:** compare the light and heavy groups, then move the fit window. For 3D diffusion, fit \(\mathrm{MSD}=st+b\) and calculate \(D=s/6\).

    **Saved data:** 25 unsmoothed samples from the completed 10,000-step Atomify run: 6,750 LJ atoms (1,125 mass 1; 5,625 mass 4), NVE, \(dt=0.005\,\tau\), initial \(T=3.00\). The single-origin group MSD removes centre-of-mass motion. The initial temperature sets velocities; the input has no thermostat or separate equilibration stage.

    The input reports \(\mathrm{MSD}/(6t)\). Treat the windowed fits as exploratory estimates from this finite mixing run, not equilibrium transport benchmarks. Units are \(\sigma^2/\tau\); SI conversion requires the Lennard-Jones length and time scales.
    """)
    return


@app.cell
def _(StringIO, np):
    csv_text = """time_tau,msd_light_sigma2,msd_heavy_sigma2
0,0,0
0.5,0.98516314,0.36222339
1,2.4593369,1.0088774
1.5,3.8604025,1.7314798
2,5.2644982,2.4716202
2.5,6.5859635,3.2653216
5,13.250133,7.3660532
7.5,19.701767,11.658152
10,26.015124,16.021536
12.5,32.401715,20.340107
15,38.092072,24.719117
17.5,44.320973,29.308171
20,50.397899,33.430243
22.5,56.165346,37.914852
25,62.316312,42.359837
27.5,68.328285,46.891092
30,73.972866,50.989508
32.5,80.09397,55.47067
35,85.839811,60.163671
37.5,90.246957,64.508441
40,96.491356,68.865081
42.5,101.53733,73.635528
45,108.05057,78.187656
47.5,112.94753,82.738049
50,118.94739,87.702142
"""
    saved_msd = np.genfromtxt(
        StringIO(csv_text),
        delimiter=",",
        names=True,
    )
    lag_time_tau = saved_msd["time_tau"]
    msd_by_group_sigma2 = {
        "Light atoms (mass 1)": saved_msd["msd_light_sigma2"],
        "Heavy atoms (mass 4)": saved_msd["msd_heavy_sigma2"],
    }
    return lag_time_tau, msd_by_group_sigma2


@app.cell(hide_code=True)
def _(mo):
    species_dropdown = mo.ui.dropdown(
        options=["Light atoms (mass 1)", "Heavy atoms (mass 4)"],
        value="Light atoms (mass 1)",
        label="atom group",
    )
    fit_start_slider = mo.ui.slider(
        start=0.0,
        stop=30.0,
        step=2.5,
        value=10.0,
        label="fit start t (τ)",
        show_value=True,
    )
    fit_end_slider = mo.ui.slider(
        start=35.0,
        stop=50.0,
        step=2.5,
        value=50.0,
        label="fit end t (τ)",
        show_value=True,
    )
    mo.hstack(
        [
            species_dropdown,
            fit_start_slider,
            fit_end_slider,
        ],
        align="center",
    )
    return fit_end_slider, fit_start_slider, species_dropdown


@app.cell
def _(
    fit_end_slider,
    fit_start_slider,
    lag_time_tau,
    mo,
    msd_by_group_sigma2,
    np,
    plt,
    species_dropdown,
):
    selected_msd_sigma2 = msd_by_group_sigma2[species_dropdown.value]
    fit_start_tau = fit_start_slider.value
    fit_end_tau = fit_end_slider.value
    fit_mask = (
        (lag_time_tau >= fit_start_tau)
        & (lag_time_tau <= fit_end_tau)
    )
    fit_slope, fit_intercept = np.polyfit(
        lag_time_tau[fit_mask],
        selected_msd_sigma2[fit_mask],
        deg=1,
    )
    fit_diffusion_sigma2_tau = fit_slope / 6
    fitted_msd_sigma2 = (
        fit_slope * lag_time_tau[fit_mask] + fit_intercept
    )

    positive_time_mask = lag_time_tau > 0
    instantaneous_diffusion = (
        selected_msd_sigma2[positive_time_mask]
        / (6 * lag_time_tau[positive_time_mask])
    )
    positive_lag_tau = lag_time_tau[positive_time_mask]
    first_ratio_index = np.argmin(np.abs(positive_lag_tau - 0.5))
    first_ratio = instantaneous_diffusion[first_ratio_index]
    last_ratio = instantaneous_diffusion[-1]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.8),
        sharex=True,
        layout="constrained",
    )
    axes[0].plot(
        lag_time_tau,
        selected_msd_sigma2,
        "o-",
        color="#0072B2",
        lw=1.7,
        ms=4,
        label=f"{species_dropdown.value}: saved MSD",
    )
    axes[0].plot(
        lag_time_tau[fit_mask],
        fitted_msd_sigma2,
        "--",
        color="#D55E00",
        lw=2.4,
        label=f"linear fit: D = {fit_diffusion_sigma2_tau:.3f} σ²/τ",
    )
    axes[0].axvspan(
        fit_start_tau,
        fit_end_tau,
        color="#D55E00",
        alpha=0.10,
        label="selected fit window",
    )
    axes[0].set(
        ylabel="MSD (σ²)",
        title="Windowed fit to actual Atomify MSD samples",
    )
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(
        positive_lag_tau,
        instantaneous_diffusion,
        "o-",
        color="#6A3D9A",
        lw=1.7,
        ms=4,
        label="MSD / (6t)",
    )
    axes[1].axhline(
        fit_diffusion_sigma2_tau,
        color="#D55E00",
        lw=2,
        ls="--",
        label="slope-fit D",
    )
    axes[1].axvspan(
        fit_start_tau,
        fit_end_tau,
        color="#D55E00",
        alpha=0.10,
    )
    axes[1].set(
        xlabel="lag time t (τ)",
        ylabel="diffusion estimate (σ²/τ)",
        title="The single-lag ratio varies with lag time",
    )
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.25)

    mo.vstack(
        [
            fig,
            mo.md(
                rf"""
                **Fit result:** \(D = \text{{slope}}/6 = {fit_diffusion_sigma2_tau:.4f}\ \sigma^2/\tau\)
                from {int(np.sum(fit_mask))} saved points over
                \(t={fit_start_tau:.1f}\)–{fit_end_tau:.1f}\ \tau\).
                The fitted slope is {fit_slope:.4f} \(\sigma^2/\tau\), with
                intercept {fit_intercept:.3f} \(\sigma^2\).
                At \(t=0.5\ \tau\), MSD/(6t) is {first_ratio:.4f}; at the last saved
                lag it is {last_ratio:.4f} \(\sigma^2/\tau\).
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

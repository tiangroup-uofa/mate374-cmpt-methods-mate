# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9", "ase>=3.26", "weas-widget>=0.2.6"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import time
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from ase import Atoms
    from weas_widget import WeasWidget

    return Atoms, WeasWidget, minimize, mo, np, plt, time


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    ## L16 · The most stable shape of an argon cluster

    **Live-coding notebook.** Each cell marked **Live step** holds one stage of
    the program: build a cluster as an $N\times3$ matrix, evaluate its energy
    and gradient with NumPy broadcasting, relax it with a multidimensional
    minimizer, and diagonalize its Hessian to test stability and obtain
    vibrational frequencies. Step 5 repeats the relaxation from many random
    starts to compare local and global minima, and step 6 uses the Hessian
    eigenvalues of two competing minima to compare their free energies
    $F=E-TS$ at finite temperature.

    **Units.** The calculation uses Lennard-Jones (LJ) reduced units, in which
    energy is measured in $\varepsilon$, length in $\sigma$, and mass in the
    atomic mass $m$. The pair energy is then $V(r)=4(r^{-12}-r^{-6})$. We convert
    to argon units ($\varepsilon=0.0103$ eV, $\sigma=3.40$ Å, $m=39.948$ u, as in
    Assignment 1) only when we report results.
    """)
    return


@app.cell(hide_code=True)
def supplied_constants(np):
    # Argon parameters used in Assignment 1, and SI constants for unit conversion.
    EPS_EV, SIGMA_A, MASS_U = 0.0103, 3.40, 39.948
    _eV, _u, _c = 1.602176634e-19, 1.66053906660e-27, 2.99792458e10  # J, kg, cm/s
    # Angular frequency unit sqrt(eps/(m sigma^2)) in rad/s, then wavenumber per unit sqrt(eigenvalue).
    OMEGA_UNIT = np.sqrt(EPS_EV*_eV/(MASS_U*_u*(SIGMA_A*1e-10)**2))
    CM1_PER_UNIT = OMEGA_UNIT/(2*np.pi*_c)

    # Lowest known energies (units of epsilon), Cambridge Cluster Database (Wales and Doye, 1997).
    REFERENCE_ENERGY = {
        2: -1.0, 3: -3.0, 4: -6.0, 5: -9.103852, 6: -12.712062, 7: -16.505384,
        8: -19.821489, 9: -24.113360, 10: -28.422532, 11: -32.765970, 12: -37.967600,
        13: -44.326801, 14: -47.845157, 15: -52.322627, 16: -56.815742, 17: -61.317995,
        18: -66.530949, 19: -72.659782, 20: -77.177043, 26: -108.315616, 38: -173.928427,
        55: -279.248470,
    }
    return CM1_PER_UNIT, EPS_EV, REFERENCE_ENERGY, SIGMA_A


@app.cell(hide_code=True)
def step_one_text(mo):
    mo.md(r"""
    ### Live step 1 · A cluster is an $N\times3$ matrix

    Row $i$ of `X` holds the coordinates $(x_i, y_i, z_i)$ of atom $i$. To build a
    compact starting structure, cut a ball out of a face-centred cubic (fcc)
    crystal: generate integer lattice points $(i,j,k)$, keep those with an even
    sum $i+j+k$, scale them by half the cube edge $a/2$, and keep the $N$ points
    closest to a chosen centre.

    - Centre `"atom"` puts a lattice site at the origin. For $N=13$ this gives a
      **cuboctahedron**: one atom and its twelve nearest neighbours.
    - Centre `"hole"` uses the octahedral hole at $(a/2,0,0)$. For $N=38$ this gives
      a **truncated octahedron**.

    With nearest-neighbour distance $r_m=2^{1/6}$ (the minimum of the pair
    potential), the cube edge is $a=\sqrt2\,r_m$.

    ```python
    g = np.arange(-4, 5)
    ijk = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T   # (729, 3) integer triples
    order = np.argsort(distances, kind="stable")             # nearest first
    ```
    """)
    return


@app.cell
def live_build(np):
    # Live step 1: return the N fcc sites closest to the chosen centre as an (N, 3) array.
    def fcc_cluster(N, centre="hole"):
        r_m = 2**(1/6)
        a = np.sqrt(2)*r_m
        g = np.arange(-4, 5)
        ijk = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
        sites = ijk[ijk.sum(axis=1) % 2 == 0]*a/2
        origin = np.array([a/2, 0.0, 0.0]) if centre == "hole" else np.zeros(3)
        distances = np.linalg.norm(sites - origin, axis=1)
        X = sites[np.argsort(distances, kind="stable")[:N]]
        return X - X.mean(axis=0)

    return (fcc_cluster,)


@app.cell(hide_code=True)
def supplied_random(np):
    def random_cluster(N, seed=0, min_distance=0.9):
        """Random positions in a sphere of reduced density 0.5, no pair closer than min_distance."""
        rng = np.random.default_rng(seed)
        radius = (3*N/(4*np.pi*0.5))**(1/3)
        X = np.empty((0, 3))
        while len(X) < N:
            trial = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(trial) < radius and np.all(np.linalg.norm(X - trial, axis=1) > min_distance):
                X = np.vstack([X, trial])
        return X - X.mean(axis=0)

    return (random_cluster,)


@app.cell(hide_code=True)
def supplied_viewer(Atoms, SIGMA_A, WeasWidget, mo, np):
    def draw_clusters(structures, titles):
        """Show reduced-unit coordinates as argon atoms (Å) in side-by-side WEAS viewers."""
        panels = []
        for X, title in zip(structures, titles):
            positions = SIGMA_A*(X - X.mean(axis=0))
            extent = max(1.0, float(np.linalg.norm(positions, axis=1).max()))
            viewer = WeasWidget(
                from_ase=Atoms("Ar"*len(X), positions=positions, pbc=False),
                viewerStyle={"width": "100%", "height": "340px"},
                modelStyle=1,
                cameraSetting={"lookAt": [0, 0, 0], "direction": [1, 0.6, 0.4],
                               "distance": 4*(extent + 2), "zoom": 1},
                cellSettings={"showCell": False},
            )
            viewer.avr.bond.add_bond_pair("Ar", max=1.25*2**(1/6)*SIGMA_A,
                                          color1="#888888", color2="#888888")
            # WeasWidget wraps its AnyWidget canvas in an ipywidgets HBox; pass the canvas to marimo.
            panels.append(mo.vstack([mo.md(f"**{title}**"), mo.ui.anywidget(viewer.children[0])]))
        return mo.vstack([
            mo.hstack(panels, widths="equal", align="start"),
            mo.md("Drag to rotate and scroll to zoom. Grey sticks join neighbours within "
                  "1.25 r<sub>m</sub> for display only; every pair contributes to the energy."),
        ])

    return (draw_clusters,)


@app.cell(hide_code=True)
def cluster_controls(mo):
    n_atoms = mo.ui.slider(3, 60, value=38, step=1, label="Number of atoms N",
                           show_value=True, debounce=True)
    start_kind = mo.ui.dropdown(
        {"fcc fragment, centred on an octahedral hole": "hole",
         "fcc fragment, centred on an atom": "atom",
         "random positions": "random"},
        value="fcc fragment, centred on an octahedral hole", label="Starting structure")
    start_seed = mo.ui.number(start=0, stop=999, step=1, value=0, label="Random seed")
    mo.vstack([
        mo.md("### Choose a starting structure"),
        mo.hstack([n_atoms, start_kind, start_seed], justify="start", gap=2),
        mo.md("Try **N = 38** from the octahedral hole, then **N = 13** centred on an atom. "
              "The random seed only affects random starts. Atom counts above about 60 make "
              "the multistart search in step 5 slow in the browser."),
    ])
    return n_atoms, start_kind, start_seed


@app.cell(hide_code=True)
def starting_structure(fcc_cluster, n_atoms, random_cluster, start_kind, start_seed):
    N = n_atoms.value
    if start_kind.value == "random":
        X_start = random_cluster(N, seed=int(start_seed.value))
    else:
        X_start = fcc_cluster(N, centre=start_kind.value)
    return N, X_start


@app.cell(hide_code=True)
def step_two_text(mo):
    mo.md(r"""
    ### Live step 2 · Energy and gradient without loops

    Broadcasting `X[:, None, :] - X[None, :, :]` subtracts every row from every
    other row and returns an $N\times N\times3$ array of separation vectors
    $\mathbf d_{ij}=\mathbf x_i-\mathbf x_j$. Its norm along the last axis is the
    $N\times N$ distance matrix $r_{ij}$. The energy sums each pair once,

    $$
    E=\sum_{i<j}4\left(r_{ij}^{-12}-r_{ij}^{-6}\right),
    $$

    using the upper triangle `np.triu_indices(N, k=1)`. The gradient with respect
    to atom $i$ adds one contribution from every partner $j$:

    $$
    \frac{\partial E}{\partial\mathbf x_i}=\sum_{j\ne i}\frac{V'(r_{ij})}{r_{ij}}\,\mathbf d_{ij},
    \qquad V'(r)=-\frac{48}{r^{13}}+\frac{24}{r^{7}}.
    $$

    The force on atom $i$ is the negative of this gradient. Setting the diagonal
    of the distance matrix to infinity removes the self-interaction $j=i$.
    """)
    return


@app.cell
def live_energy(np):
    # Live step 2: vectorized LJ energy and gradient in reduced units. X has shape (N, 3).
    def pair_separations(X):
        d = X[:, None, :] - X[None, :, :]        # (N, N, 3) separation vectors
        r = np.linalg.norm(d, axis=-1)           # (N, N) distance matrix
        return d, r

    def lj_energy(X):
        d, r = pair_separations(X)
        i, j = np.triu_indices(len(X), k=1)
        inv6 = r[i, j]**-6
        return np.sum(4*(inv6*inv6 - inv6))

    def lj_gradient(X):
        d, r = pair_separations(X)
        np.fill_diagonal(r, np.inf)
        inv6 = r**-6
        dV_dr = (-48*inv6*inv6 + 24*inv6)/r
        return np.sum((dV_dr/r)[:, :, None]*d, axis=1)   # (N, 3)

    return lj_energy, lj_gradient


@app.cell(hide_code=True)
def check_gradient(X_start, lj_energy, lj_gradient, mo, np):
    def finite_difference_check(X, h=1e-6, samples=6):
        """Compare a few gradient entries with central differences of the energy."""
        x0 = X.ravel()
        picks = np.linspace(0, x0.size - 1, samples).astype(int)
        worst = 0.0
        for k in picks:
            step = np.zeros_like(x0)
            step[k] = h
            numeric = (lj_energy((x0 + step).reshape(-1, 3)) - lj_energy((x0 - step).reshape(-1, 3)))/(2*h)
            worst = max(worst, abs(numeric - lj_gradient(X).ravel()[k]))
        return worst

    gradient_error = finite_difference_check(X_start)
    E_start = lj_energy(X_start)
    gradient_ok = gradient_error < 1e-5
    mo.vstack([
        mo.md(f"Starting energy **E = {E_start:.6f} ε** for N = {len(X_start)} atoms "
              f"({len(X_start)*(len(X_start)-1)//2} pairs)."),
        mo.callout(f"Step 2 check passed: gradient agrees with finite differences to {gradient_error:.1e}."
                   if gradient_ok else
                   f"The gradient differs from finite differences by {gradient_error:.2e}: check signs and the factor 1/r.",
                   kind="success" if gradient_ok else "warn"),
    ])
    return E_start, gradient_ok


@app.cell(hide_code=True)
def step_three_text(mo):
    mo.md(r"""
    ### Live step 3 · Relax all $3N$ coordinates with `minimize`

    The minimizer works with one flat vector $\mathbf x=(x_1,y_1,z_1,\ldots,z_N)$
    of length $3N$, so the objective reshapes it to $N\times3$ before calling our
    functions and flattens the gradient again. L-BFGS-B uses the gradient to build
    an approximate Hessian from recent steps and stops when the gradient is small.

    ```python
    result = minimize(fun, x0, jac=grad, method="L-BFGS-B",
                      options={"gtol": 1e-6, "ftol": 1e-12, "maxiter": 5000})
    result.x, result.fun, result.nit, result.success
    ```

    Docs: [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    """)
    return


@app.cell
def live_relax(lj_energy, lj_gradient, minimize):
    # Live step 3: relax a structure. Return the relaxed (N, 3) matrix and the SciPy result.
    def relax(X0):
        N = len(X0)
        result = minimize(lambda x: lj_energy(x.reshape(N, 3)), X0.ravel(),
                          jac=lambda x: lj_gradient(x.reshape(N, 3)).ravel(),
                          method="L-BFGS-B",
                          options={"gtol": 1e-6, "ftol": 1e-12, "maxiter": 5000})
        return result.x.reshape(N, 3), result

    return (relax,)


@app.cell(hide_code=True)
def run_relax(N, REFERENCE_ENERGY, X_start, lj_gradient, relax, time):
    _t0 = time.perf_counter()
    X_min, relax_result = relax(X_start)
    relax_seconds = time.perf_counter() - _t0
    E_min = relax_result.fun
    max_gradient = abs(lj_gradient(X_min)).max()
    E_reference = REFERENCE_ENERGY.get(N)
    return E_min, E_reference, X_min, max_gradient, relax_result, relax_seconds


@app.cell(hide_code=True)
def show_relax(E_min, E_reference, E_start, EPS_EV, N, X_min, X_start, draw_clusters, max_gradient, mo, relax_result, relax_seconds):
    _reference = ("no reference value stored for this N" if E_reference is None else
                  f"lowest known minimum {E_reference:.6f} ε, difference {E_min - E_reference:+.6f} ε")
    mo.vstack([
        mo.md(f"""
    | | Start | Relaxed |
    |---|---:|---:|
    | Energy (ε) | {E_start:.6f} | **{E_min:.6f}** |
    | Energy per atom (meV) | {1000*EPS_EV*E_start/N:.3f} | {1000*EPS_EV*E_min/N:.3f} |

    {relax_result.nit} iterations in {1000*relax_seconds:.0f} ms; largest gradient entry {max_gradient:.1e};
    `success = {relax_result.success}`. Reference: {_reference}.
    """),
        draw_clusters([X_start, X_min], ["Starting structure", "After minimization"]),
    ])
    return


@app.cell(hide_code=True)
def step_four_text(mo):
    mo.md(r"""
    ### Live step 4 · The Hessian and its eigenvalues

    A zero gradient identifies a **stationary point**: a minimum, a maximum, or a
    saddle. The second derivatives decide which. The Hessian is the symmetric
    $3N\times3N$ matrix $H_{kl}=\partial^2E/\partial x_k\,\partial x_l$. Column $l$
    is the change in the gradient when coordinate $l$ moves, so a central
    difference of our analytical gradient builds one column at a time:

    $$
    H_{:,l}\approx\frac{\nabla E(\mathbf x+h\,\mathbf e_l)-\nabla E(\mathbf x-h\,\mathbf e_l)}{2h}.
    $$

    `np.linalg.eigh` returns the eigenvalues $\lambda_k$ in ascending order and the
    eigenvectors (modes) as columns. For an isolated cluster, six eigenvalues are
    zero: three rigid translations and three rigid rotations leave the energy
    unchanged. At a minimum, the other $3N-6$ are positive, and each gives a
    harmonic vibrational frequency $\omega_k=\sqrt{\lambda_k/m}$. A negative
    eigenvalue means the energy decreases along that mode: the structure is a
    saddle point.

    Docs: [`numpy.linalg.eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html)
    """)
    return


@app.cell
def live_hessian(lj_gradient, np):
    # Live step 4: finite-difference Hessian from the analytical gradient, then its eigenpairs.
    def hessian(X, h=1e-5):
        x0 = X.ravel()
        n = x0.size
        H = np.empty((n, n))
        for l in range(n):
            step = np.zeros(n)
            step[l] = h
            H[:, l] = (lj_gradient((x0 + step).reshape(-1, 3)).ravel()
                       - lj_gradient((x0 - step).reshape(-1, 3)).ravel())/(2*h)
        return 0.5*(H + H.T)   # remove round-off asymmetry

    def normal_modes(X):
        eigenvalues, modes = np.linalg.eigh(hessian(X))
        return eigenvalues, modes

    return (normal_modes,)


@app.cell(hide_code=True)
def supplied_classify(np):
    def classify(eigenvalues, tol=1e-3):
        """Count negative, near-zero, and positive Hessian eigenvalues (reduced units)."""
        negative = int(np.sum(eigenvalues < -tol))
        zero = int(np.sum(np.abs(eigenvalues) <= tol))
        positive = int(np.sum(eigenvalues > tol))
        if negative == 0 and zero == 6:
            kind = "a local minimum"
        elif negative > 0:
            kind = f"a saddle point with {negative} downhill direction(s)"
        else:
            kind = f"undetermined: {zero} near-zero eigenvalues instead of 6"
        return negative, zero, positive, kind

    return (classify,)


@app.cell(hide_code=True)
def show_modes(CM1_PER_UNIT, N, X_min, classify, mo, normal_modes, np, plt, time):
    _t0 = time.perf_counter()
    eigenvalues, modes = normal_modes(X_min)
    hessian_seconds = time.perf_counter() - _t0
    n_negative, n_zero, n_positive, stationary_kind = classify(eigenvalues)

    def plot_spectrum():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
        colours = np.where(eigenvalues < -1e-3, "tab:red",
                           np.where(np.abs(eigenvalues) <= 1e-3, "tab:gray", "tab:blue"))
        ax1.scatter(np.arange(len(eigenvalues)), eigenvalues, c=colours, s=14)
        ax1.axhline(0, color="black", lw=0.8)
        ax1.set(xlabel="Mode index k (ascending)", ylabel="Eigenvalue λ (ε/σ²)",
                title=f"Hessian eigenvalues, 3N = {3*N}")
        wavenumbers = CM1_PER_UNIT*np.sqrt(eigenvalues[eigenvalues > 1e-3])
        ax2.hist(wavenumbers, bins=25, color="tab:blue", alpha=0.8)
        ax2.set(xlabel="Harmonic wavenumber (cm⁻¹), argon units", ylabel="Number of modes",
                title="Vibrational frequencies")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    _vib = CM1_PER_UNIT*np.sqrt(eigenvalues[eigenvalues > 1e-3])
    _range = (f"Vibrational wavenumbers span {_vib.min():.1f}–{_vib.max():.1f} cm⁻¹ "
              f"({0.0299792458*_vib.min():.2f}–{0.0299792458*_vib.max():.2f} THz)." if len(_vib) else "")
    spectrum_figure = plot_spectrum()
    mo.vstack([
        spectrum_figure,
        mo.md(f"{n_negative} negative, {n_zero} near-zero, and {n_positive} positive eigenvalues "
              f"(Hessian and diagonalization: {1000*hessian_seconds:.0f} ms). "
              f"The relaxed structure is **{stationary_kind}**. {_range}"),
    ])
    return eigenvalues, modes, n_negative, spectrum_figure


@app.cell(hide_code=True)
def saddle_escape(E_min, X_min, classify, draw_clusters, eigenvalues, lj_energy, mo, modes, n_negative, normal_modes, relax):
    mo.stop(n_negative == 0, mo.md(
        "*No negative eigenvalue: there is no downhill direction to follow. "
        "Choose N = 13 or N = 55 with the atom-centred fcc start to see a saddle point.*"))

    def escape_saddles(X, lowest_mode, max_escapes=5, step=0.1):
        """Displace along the most negative mode and relax again until no negative mode remains."""
        energies = [lj_energy(X)]
        for _ in range(max_escapes):
            X, _result = relax(X + step*lowest_mode.reshape(-1, 3))
            energies.append(lj_energy(X))
            _values, _vectors = normal_modes(X)
            if classify(_values)[0] == 0:
                break
            lowest_mode = _vectors[:, 0]
        return X, energies

    X_escaped, escape_energies = escape_saddles(X_min, modes[:, 0])
    escaped_kind = classify(normal_modes(X_escaped)[0])[3]
    _path = " → ".join(f"{e:.6f}" for e in escape_energies)
    mo.vstack([
        mo.md(f"""
    ### Following the downhill mode

    The most negative eigenvalue is λ = {eigenvalues[0]:.3f} ε/σ². Displacing every atom
    by 0.1σ along its eigenvector and minimizing again lowers the energy. If the new
    structure still has a negative eigenvalue, we repeat the step along its new
    downhill mode. Energies (ε): {_path}. The final structure is **{escaped_kind}**.
    """),
        draw_clusters([X_min, X_escaped], ["Saddle point", "After following the downhill mode"]),
    ])
    return X_escaped, escape_energies, escaped_kind


@app.cell(hide_code=True)
def step_five_text(mo):
    mo.md(r"""
    ### Live step 5 · Many random starts: local and global minima

    A minimizer goes downhill to the nearest minimum it can reach. Each random start
    may end in a different local minimum, so we relax many starts and compare the
    distribution of final energies with the lowest known energy. Press the button to
    run the search. The total time grows with both $N$ and the number of starts.
    """)
    return


@app.cell(hide_code=True)
def multistart_controls(mo):
    n_starts = mo.ui.slider(5, 40, value=10, step=5, label="Number of random starts",
                            show_value=True)
    run_search = mo.ui.run_button(label="Run multistart search")
    mo.hstack([n_starts, run_search], justify="start", gap=2)
    return n_starts, run_search


@app.cell
def live_multistart(random_cluster, relax):
    # Live step 5: relax n_starts random clusters; return the final energies and structures.
    def multistart(N, n_starts):
        energies, structures = [], []
        for seed in range(n_starts):
            X, result = relax(random_cluster(N, seed=seed))
            energies.append(result.fun)
            structures.append(X)
        return energies, structures

    return (multistart,)


@app.cell(hide_code=True)
def show_multistart(E_min, E_reference, N, X_min, draw_clusters, mo, multistart, n_starts, np, plt, run_search, time):
    mo.stop(not run_search.value, mo.md("*Press **Run multistart search** to relax the random starts.*"))
    _t0 = time.perf_counter()
    start_energies, start_structures = multistart(N, n_starts.value)
    search_seconds = time.perf_counter() - _t0
    best = int(np.argmin(start_energies))

    def plot_energies():
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.hist(start_energies, bins=20, color="tab:blue", alpha=0.8, label="Random starts")
        ax.axvline(E_min, color="tab:orange", lw=2, label=f"Structure from step 3: {E_min:.4f} ε")
        if E_reference is not None:
            ax.axvline(E_reference, color="black", ls="--", label=f"Lowest known: {E_reference:.4f} ε")
        ax.set(xlabel="Final energy (ε)", ylabel="Number of starts",
               title=f"N = {N}: {len(start_energies)} relaxations from random positions")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    multistart_figure = plot_energies()
    mo.vstack([
        multistart_figure,
        mo.md(f"Lowest energy from random starts: **{start_energies[best]:.6f} ε** (seed {best}). "
              f"{len(start_energies)} relaxations took {search_seconds:.1f} s, "
              f"or {1000*search_seconds/len(start_energies):.0f} ms per start."),
        draw_clusters([start_structures[best], X_min],
                      ["Best of the random starts", "Structure from step 3"]),
    ])
    return multistart_figure, start_energies, start_structures


@app.cell(hide_code=True)
def landscape_text(mo):
    mo.md(r"""
    ### What the 38-atom cluster shows

    From the octahedral-hole start, a single relaxation reaches the truncated
    octahedron at $-173.928427\,\varepsilon$, the lowest known LJ$_{38}$ energy. It
    is a small piece of the fcc crystal. Random starts usually end in disordered or
    icosahedral-like minima. The lowest icosahedral structure lies only
    $0.676\,\varepsilon$ higher, but a large barrier separates the two families.
    This **double-funnel** landscape (Doye, Miller and Wales, *J. Chem. Phys.*
    **110**, 6896, 1999) makes LJ$_{38}$ a standard test for global optimization.
    """)
    return


@app.cell(hide_code=True)
def step_six_text(mo):
    mo.md(r"""
    ### Live step 6 · Which structure wins at finite temperature?

    At temperature $T$, the cluster vibrates inside its basin. In the harmonic
    approximation, each of the $3N-6$ modes is an independent oscillator, and a
    classical oscillator of angular frequency $\omega$ contributes
    $k_BT\ln(\hbar\omega/k_BT)$ to the free energy. For one minimum,

    $$
    F(T)=E_{\min}+k_BT\sum_{k=7}^{3N}\ln\frac{\hbar\omega_k}{k_BT},
    \qquad \omega_k=\sqrt{\lambda_k/m}.
    $$

    The sum skips the six zero modes. In reduced units ($k_B=1$, $T$ in
    $\varepsilon/k_B$), argon has $\hbar=0.0297$. Comparing two minima,
    $\Delta F=\Delta E-T\Delta S$, and $\hbar$ cancels: the entropy difference
    depends only on the ratio of the frequencies,

    $$
    \Delta S_{\text{ico}-\text{TO}}=k_B\sum_k\ln\frac{\omega_k^{\text{TO}}}{\omega_k^{\text{ico}}}
    =\frac{k_B}{2}\left(\sum_k\ln\lambda_k^{\text{TO}}-\sum_k\ln\lambda_k^{\text{ico}}\right).
    $$

    A softer basin (smaller eigenvalues) is wider, so it holds more entropy.
    The notebook supplies the lowest icosahedral LJ$_{38}$ structure.
    """)
    return


@app.cell(hide_code=True)
def supplied_icosahedral(np):
    # Lowest icosahedral-funnel LJ38 minimum (C5v, E = -173.252378 eps), reduced units,
    # located by basin hopping and rotated to its principal axes.
    X_icosahedral = np.array([
        [0.094671, -0.532485, 0.762604],
        [-0.470745, 0.000000, 0.000000],
        [-0.433987, 1.478510, 0.449897],
        [0.509096, 0.541338, -1.767214],
        [-0.437484, -1.089122, 1.559798],
        [0.094671, -0.017458, -0.929946],
        [-0.433987, -1.460582, 0.505072],
        [-0.437484, 1.797945, -0.621733],
        [-0.433987, 0.029008, 1.545172],
        [0.631628, -0.000000, -0.000000],
        [-1.008361, 0.899082, -0.310904],
        [-0.437484, -1.820013, -0.553813],
        [0.094671, 0.879036, -0.303973],
        [1.051607, 0.562952, -0.806238],
        [-0.437484, 1.146899, 1.517821],
        [0.509096, -0.534763, 1.769215],
        [0.509096, -1.476694, 1.111515],
        [0.094671, -0.889826, -0.270766],
        [0.094671, 0.560732, 0.742081],
        [0.509096, 1.848003, -0.031256],
        [0.509096, -1.847874, 0.038127],
        [1.051607, 0.018457, 0.983155],
        [-0.433987, 0.884762, -1.267121],
        [-0.437484, -0.035708, -1.902073],
        [0.509096, -0.607286, -1.745651],
        [1.051607, -0.592816, -0.784541],
        [-1.008361, -0.910117, -0.276940],
        [-0.433987, -0.931697, -1.233021],
        [-1.008361, -0.544627, 0.779994],
        [1.051607, 0.940740, 0.286258],
        [0.509096, -1.513438, -1.060943],
        [0.509096, 1.517372, 1.055308],
        [-1.008361, -0.017856, -0.951152],
        [0.509096, 1.472551, -1.116999],
        [-1.008361, 0.573519, 0.759003],
        [1.051607, -0.929333, 0.321365],
        [0.509096, 0.600790, 1.747897],
        [-1.584074, 0.000000, 0.000000],
    ])
    return (X_icosahedral,)


@app.cell
def live_free_energy(np):
    # Live step 6: classical harmonic free energy of one minimum, for scalar or array T.
    HBAR_ARGON = 0.0297   # hbar in LJ reduced units for argon

    def harmonic_free_energy(T, E, eigenvalues, hbar=HBAR_ARGON):
        omega = np.sqrt(eigenvalues[6:])                 # skip the six zero modes
        T = np.asarray(T, dtype=float)
        return E + T*(np.sum(np.log(hbar*omega)) - len(omega)*np.log(T))

    return (harmonic_free_energy,)


@app.cell(hide_code=True)
def compare_minima(EPS_EV, X_icosahedral, classify, draw_clusters, fcc_cluster, harmonic_free_energy, lj_energy, mo, normal_modes, np, plt, relax):
    KB_EV = 8.617333262e-5
    T_UNIT_K = EPS_EV/KB_EV                    # 1 reduced temperature unit in K for argon
    X_TO, _ = relax(fcc_cluster(38, centre="hole"))
    X_ico, _ = relax(X_icosahedral)
    lam_TO, lam_ico = normal_modes(X_TO)[0], normal_modes(X_ico)[0]
    E_TO, E_ico = lj_energy(X_TO), lj_energy(X_ico)
    delta_E = E_ico - E_TO
    delta_S = 0.5*(np.sum(np.log(lam_TO[6:])) - np.sum(np.log(lam_ico[6:])))
    T_cross = delta_E/delta_S

    def plot_free_energy():
        T = np.linspace(0.005, 0.5, 300)
        dF = harmonic_free_energy(T, E_ico, lam_ico) - harmonic_free_energy(T, E_TO, lam_TO)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
        ax1.plot(T, dF, color="black", lw=2)
        ax1.axhline(0, color="gray", lw=0.8)
        ax1.axvline(T_cross, color="tab:red", ls="--", label=f"Harmonic crossover T* = {T_cross:.3f} ε/k_B")
        ax1.text(0.02, 0.2*delta_E, "Truncated octahedron\nfavoured", fontsize=8)
        ax1.text(0.33, -0.5*delta_E, "Icosahedral\nfavoured", fontsize=8)
        ax1.set(xlabel="Temperature (ε/k_B)", ylabel="F_ico − F_TO (ε)")
        top = ax1.secondary_xaxis("top", functions=(lambda t: t*T_UNIT_K, lambda k: k/T_UNIT_K))
        top.set_xlabel("Argon temperature (K)")
        ax1.legend(fontsize=7, loc="lower left")
        k = np.arange(1, len(lam_TO) - 5)
        ax2.plot(k, np.sqrt(lam_TO[6:]), color="tab:blue", label="Truncated octahedron")
        ax2.plot(k, np.sqrt(lam_ico[6:]), color="tab:orange", label="Icosahedral minimum")
        ax2.set(xlabel="Vibrational mode (ascending)", ylabel="ω (reduced units)",
                title="Sorted vibrational frequencies")
        ax2.legend(fontsize=8)
        for ax in (ax1, ax2):
            ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig

    free_energy_figure = plot_free_energy()
    _checks = [classify(lam_TO)[3], classify(lam_ico)[3]]
    mo.vstack([
        free_energy_figure,
        mo.md(f"""
    | | Truncated octahedron | Icosahedral minimum |
    |---|---:|---:|
    | Energy (ε) | {E_TO:.6f} | {E_ico:.6f} |
    | Stationary point | {_checks[0]} | {_checks[1]} |
    | Σ ln λ over 108 modes | {np.sum(np.log(lam_TO[6:])):.3f} | {np.sum(np.log(lam_ico[6:])):.3f} |

    ΔE = **{delta_E:.3f} ε** and ΔS = **{delta_S:.2f} k<sub>B</sub>**, so the free energies cross at
    T* = ΔE/ΔS = **{T_cross:.3f} ε/k<sub>B</sub> ≈ {T_cross*T_UNIT_K:.0f} K** for argon.
    """),
        draw_clusters([X_TO, X_ico], ["Truncated octahedron (O_h)", "Lowest icosahedral minimum (C_5v)"]),
    ])
    return T_cross, delta_E, delta_S, free_energy_figure


if __name__ == "__main__":
    app.run()

"""Reproduce A2 draft numerical answers; no quantum data are assumed."""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar, minimize_scalar, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'assignments/A2/lj-data.csv'


def calculate_LJ(r, epsilon, sigma):
    q = (sigma / np.asarray(r))**6
    return 4 * epsilon * (q*q - q)


def calculate_cluster_energy(coords, epsilon, sigma):
    coords = np.asarray(coords, dtype=float)
    total = 0.0
    for i in range(len(coords)):
        for j in range(i):
            total += calculate_LJ(np.linalg.norm(coords[i] - coords[j]), epsilon, sigma)
    return float(total)


def cluster_residual(parameters, clusters):
    epsilon, sigma = parameters
    total = 0.0
    for cluster in clusters:
        coords = cluster['coords']
        n = len(coords)
        predicted = calculate_cluster_energy(coords, epsilon, sigma)
        total += ((cluster['energy_eV'] - predicted) / n)**2
    return total


def main():
    H = 2.5
    def F(x):
        return np.log(x / (1-x)) + H*(1-2*x)
    def g(x):
        return 1 / (1 + np.exp(H*(1-2*x)))
    x = 0.9
    rows = []
    for step in range(1, 1001):
        new = g(x)
        rows.append([step, float(new)])
        if abs(new-x) < 1e-4:
            break
        x = new
    else:
        raise AssertionError('Fixed point did not converge')
    beta = float(new)
    root = root_scalar(F, bracket=[0.6, 0.99], method='brentq', xtol=1e-12)
    assert root.converged and abs(F(beta)) < 1e-3

    # Fixed small additive noise in eV; neither the root nor minimum is sampled.
    r = np.array([3.2, 3.35, 3.5, 3.65, 4.0, 4.4, 5.0, 6.0])
    noise = np.array([0.00008, -0.00006, 0.00005, -0.00009,
                      0.00007, -0.00004, 0.00006, -0.00003])
    y = np.round(calculate_LJ(r, 0.0103, 3.4) + noise, 9)
    np.savetxt(DATA, np.column_stack([r, y]), delimiter=',',
               header='r_A,V_eV', comments='', fmt=['%.2f', '%.9f'])
    spline = CubicSpline(r, y, bc_type='natural', extrapolate=False)
    zero = root_scalar(spline, bracket=[3.35, 3.5], method='brentq')
    minimum = minimize_scalar(spline, bounds=[3.5, 4.4], method='bounded',
                              options={'xatol': 1e-12})
    epsilon0 = float(-minimum.fun)
    sigma0 = float(minimum.x / 2**(1/6))
    def objective(parameters):
        return float(np.sum((y-calculate_LJ(r, *parameters))**2))
    fit = minimize(objective, [epsilon0, sigma0], method='Nelder-Mead',
                   bounds=[(1e-6, 0.1), (2.5, 4.5)],
                   options={'xatol': 1e-11, 'fatol': 1e-18, 'maxiter': 10000})
    assert zero.converged and minimum.success and fit.success
    assert abs(fit.x[0] / 0.0103 - 1) < 0.02
    assert abs(fit.x[1] / 3.4 - 1) < 0.01

    # Q4 implementation check only: synthetic totals, not quantum results.
    test_clusters = [
        {'coords': np.array([[0., 0., 0.], [4., 0., 0.]])},
        {'coords': np.array([[0., 0., 0.], [4., 0., 0.], [0., 4.5, 0.]])},
    ]
    for cluster in test_clusters:
        cluster['energy_eV'] = calculate_cluster_energy(cluster['coords'], 0.0103, 3.4)
    assert cluster_residual([0.0103, 3.4], test_clusters) == 0.0
    assert cluster_residual([0.012, 3.5], test_clusters) > 0
    assert np.isclose(test_clusters[1]['energy_eV'], -0.017209784665, atol=1e-12)

    results = {
        'q2': {'beta': beta, 'alpha': 1-beta, 'iterations': step,
               'last_step': abs(new-x), 'F_beta': float(F(beta)),
               'slope_beta': 2*H*beta*(1-beta), 'iteration_table': rows,
               'brent_root': root.root, 'brent_iterations': root.iterations},
        'q3': {'root': zero.root, 'r_min': minimum.x, 'epsilon_interpolated': epsilon0,
               'sigma_from_min': sigma0, 'epsilon_fit': fit.x[0], 'sigma_fit': fit.x[1],
               'residual': fit.fun, 'fit_iterations': fit.nit,
               'exact_r_min': 2**(1/6)*3.4},
    }
    (HERE / 'results.json').write_text(json.dumps(results, indent=2)+'\n')
    grid = np.linspace(r[0], r[-1], 600)
    fig, ax = plt.subplots(figsize=(6.4, 3.5), layout='constrained')
    ax.scatter(r, y, color='black', s=22, label='Given data', zorder=3)
    ax.plot(grid, spline(grid), label='Natural cubic spline')
    ax.plot(grid, calculate_LJ(grid, *fit.x), label='Fitted LJ')
    ax.plot(grid, calculate_LJ(grid, 0.0103, 3.4), '--', label='Reference LJ')
    ax.axhline(0, color='gray', lw=0.6)
    ax.set(xlabel='Pair distance (Å)', ylabel='Pair energy (eV)', ylim=(-0.013, 0.028))
    ax.legend(fontsize=8)
    fig.savefig(HERE / 'lj-fit.png', dpi=300)
    plt.close(fig)
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'iteration_table'}
                      for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()

"""Reproduce the A1 answer calculations and figures.
Run: MPLBACKEND=Agg uv run answer-keys/A1/check_answers.py
"""
import ast
import json
from pathlib import Path
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
epsilon, sigma = 0.0103, 3.4


def calculate_LJ(r):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


def calculate_LJ_cluster(coords):
    total = 0.0
    for i in range(len(coords)):
        for j in range(i):
            total += calculate_LJ(np.linalg.norm(np.asarray(coords[i]) - coords[j]))
    return total


def estimate_pi(N, **parameters):
    total = 0.0
    for k in range(N):
        total += (-1.0)**k / (2*k + 1)
    return 4 * total


# Use precisely the current supplied geometry and optimizer implementations.
tree = ast.parse((ROOT / 'assignments/A1/a1_lj_clusters.edit.py').read_text())
functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
             and n.name in {'four_atom_shapes', 'random_cluster', 'optimize_cluster'}]
exec(compile(ast.Module(body=functions, type_ignores=[]), '<A1 supplied code>', 'exec'))

assert bin(2026) == '0b11111101010'
assert int('1100111', 2) == 103
assert Fraction(1, 10) - Fraction(int('00011001', 2), 256) == Fraction(3, 1280)
results = {'pi': [], 'shapes': [], 'optimizer': [], 'sampled_pair': []}
for n in [10, 100, 1000, 10000, 100000]:
    estimate = estimate_pi(n)
    error = abs(estimate - np.pi)
    results['pi'].append([n, estimate, error, error / np.pi])
fig, ax = plt.subplots(figsize=(6, 2.5), layout='constrained')
ax.loglog([r[0] for r in results['pi']], [r[2] for r in results['pi']], 'o-')
ax.set(xlabel='Retained terms N', ylabel='Absolute error', title='Leibniz convergence')
ax.grid(alpha=.2)
fig.savefig(OUT / 'pi-convergence.png', dpi=170)
plt.close(fig)

for points in [400, 2000]:
    r = np.linspace(3.1, 10, points)
    energy = calculate_LJ(r)
    bracket = int(np.flatnonzero(energy[:-1] * energy[1:] <= 0)[0])
    root = bracket + int(np.argmin(abs(energy[bracket:bracket+2])))
    best = int(np.argmin(energy))
    results['sampled_pair'].append([points, float(r[root]), float(energy[root]), float(r[best]), float(energy[best])])

lengths = np.linspace(3.2, 5, 361)
fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), layout='constrained')
r = np.linspace(3.1, 10, 400)
axes[0].plot(r, calculate_LJ(r))
axes[0].axhline(0, color='grey', lw=.7)
axes[0].set(xlabel='Pair distance (Å)', ylabel='Pair energy (eV)')
for name in four_atom_shapes(1):
    energies = np.array([calculate_LJ_cluster(four_atom_shapes(a)[name]) for a in lengths])
    best = int(np.argmin(energies))
    results['shapes'].append([name, float(lengths[best]), float(np.min(energies))])
    axes[1].plot(lengths, energies, label=name)
axes[1].set(xlabel='Spacing / edge length (Å)', ylabel='Cluster energy (eV)')
axes[1].legend(fontsize=8)
fig.savefig(OUT / 'lj-comparison.png', dpi=170)
plt.close(fig)
assert abs(calculate_LJ_cluster([[0,0,0],[4,0,0],[0,4.5,0]]) + .01720978466531022) < 1e-14
for seed in [1, 2, 3]:
    initial, final, result, history = optimize_cluster(7, seed)
    results['optimizer'].append([seed, calculate_LJ_cluster(initial), calculate_LJ_cluster(final), bool(result.success)])
    assert calculate_LJ_cluster(final) < calculate_LJ_cluster(initial)
(OUT / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))

"""Consistent L26 copper models: smooth LJ cutoff, fitted with that same cutoff."""
from pathlib import Path
import numpy as np
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.eam import EAM
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
def lj(epsilon=1., sigma=1.):
    return LennardJones(epsilon=epsilon, sigma=sigma, rc=3*sigma,
                        ro=2*sigma, smooth=True)
def crystal(structure, d):
    factors = {"fcc":2**.5, "bcc":2/3**.5, "sc":1., "diamond":4/3**.5}
    return bulk("Cu", structure, a=d*factors[structure])
def fit_lj():
    def energy(d):
        a = crystal("fcc", d); a.calc = lj()
        return a.get_potential_energy()/len(a)
    best = minimize_scalar(energy, bounds=(1.,1.2), method="bounded",
                           options={"xatol":1e-12})
    return 3.54/-best.fun, (3.615/2**.5)/best.x
EPS, SIG = fit_lj()
def calculator(name):
    if name == "LJ": return lj(EPS, SIG)
    if name == "EAM": return EAM(potential=str(ROOT/"activities/Cu_mishin1.eam.alloy"))
    raise ValueError(name)
def energy(name, structure, d):
    a = crystal(structure, d); a.calc = calculator(name)
    return a.get_potential_energy()/len(a)
def minimum(name, structure):
    return minimize_scalar(lambda d:energy(name,structure,d),
                           bounds=(2.1,3.1),method="bounded",
                           options={"xatol":1e-10})
if __name__ == "__main__":
    print("epsilon, sigma", EPS,SIG)
    for name in ("LJ","EAM"):
        for structure in ("diamond","sc","bcc","fcc"):
            b=minimum(name,structure); h=1e-3
            k=(energy(name,structure,b.x+h)-2*b.fun+energy(name,structure,b.x-h))/h**2
            print(name,structure,b.x,b.fun,k)

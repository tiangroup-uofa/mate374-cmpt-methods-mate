"""Low-energy NVE comparison: 32 periodic Cu atoms, no heating ramp."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from ase import units
from ase.build import bulk
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from scipy.optimize import minimize_scalar
from l26_models import calculator, ROOT, EPS, SIG
OUT=ROOT/"data/l26"
def run(name):
    if name=="MACE":
        from mace.calculators import mace_mp
        model=Path.home()/".cache/mace/20231203mace128L1_epoch199model"
        calc=mace_mp(model=str(model),device="cuda",default_dtype="float64")
    else:
        calc=calculator(name)
    def e(a):
        cell=bulk("Cu","fcc",a=a,cubic=True)
        cell.calc=calc
        return cell.get_potential_energy()/len(cell)
    opt=minimize_scalar(e,bounds=(3.5,3.8),method="bounded",options={"xatol":1e-7})
    initial=bulk("Cu","fcc",a=opt.x,cubic=True).repeat(2)
    MaxwellBoltzmannDistribution(initial,temperature_K=100,rng=np.random.default_rng(374))
    Stationary(initial)
    # Each model starts at its own zero-pressure lattice, with identical seeded momenta.
    report={"a_A":opt.x,"minimum_eV_per_atom":opt.fun,"steps":{}}
    for dt in (1.,.5):
        atoms=initial.copy(); atoms.calc=calc
        dyn=VelocityVerlet(atoms,dt*units.fs)
        rows=[]
        for step in range(round(500/dt)+1):
            if step%round(5/dt)==0:
                rows.append([dyn.get_time()/units.fs,atoms.get_potential_energy()/len(atoms),
                             atoms.get_kinetic_energy()/len(atoms),atoms.get_temperature()])
            if step<round(500/dt): dyn.run(1)
        arr=np.array(rows)
        drift=arr[:,1]+arr[:,2]-arr[0,1]-arr[0,2]
        report["steps"][str(dt)]={"max_total_energy_error_eV_per_atom":float(max(abs(drift)))}
        np.savetxt(OUT/f"l26-nve-{name}-{dt:g}fs.csv",arr,delimiter=",",
                   header="time_fs,potential_eV_per_atom,kinetic_eV_per_atom,temperature_K",comments="")
    if name=="MACE":
        report["model_sha256"]=hashlib.sha256(model.read_bytes()).hexdigest()
        report["parameters"]=sum(p.numel() for p in calc.models[0].parameters())
    return report
if __name__=="__main__":
    report={"protocol":{"atoms":32,"initial_temperature_K":100,"duration_fs":500,
                        "ensemble":"NVE","integrator":"velocity Verlet","seed":374,
                        "precision":"float64","LJ_epsilon_eV":EPS,"LJ_sigma_A":SIG,
                        "LJ_cutoff":"smooth, ro=2 sigma, rc=3 sigma",
                        "EAM":"Mishin Cu EAM1","MACE":"MACE-MP-0 medium"},"runs":{}}
    for name in ("LJ","EAM","MACE"):
        report["runs"][name]=run(name)
        print(name,report["runs"][name],flush=True)
        (OUT/"l26-nve-provenance.json").write_text(json.dumps(report,indent=2))

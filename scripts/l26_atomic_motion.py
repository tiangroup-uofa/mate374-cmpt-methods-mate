"""Same 32 Cu atoms and initial velocities under three actual force models."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from ase import units
from ase.build import bulk
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from l26_models import calculator, ROOT
OUT=ROOT/"data/l26"
def run():
    initial=bulk("Cu","fcc",a=3.615,cubic=True).repeat(2)
    MaxwellBoltzmannDistribution(initial,temperature_K=100,rng=np.random.default_rng(374))
    Stationary(initial)
    report={"atoms":32,"a_A":3.615,"initial_velocity_temperature_K":100,
            "seed":374,"dt_fs":0.5,"duration_fs":1000,"sample_fs":2,
            "ensemble":"NVE","precision":"float64","runs":{}}
    for name in ("LJ","EAM","MACE"):
        atoms=initial.copy()
        if name=="MACE":
            from mace.calculators import mace_mp
            checkpoint=Path.home()/".cache/mace/20231203mace128L1_epoch199model"
            atoms.calc=mace_mp(model=str(checkpoint),device="cuda",default_dtype="float64")
            report["runs"][name]={"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        else:
            atoms.calc=calculator(name);report["runs"][name]={}
        dyn=VelocityVerlet(atoms,.5*units.fs)
        positions=[]
        for step in range(2001):
            if step%4==0:positions.append(atoms.positions.copy())
            if step<2000:dyn.run(1)
        positions=np.array(positions)
        assert np.isfinite(positions).all()
        displacement=positions-positions[0]
        report["runs"][name]["max_displacement_A"]=float(np.linalg.norm(displacement,axis=-1).max())
        np.savez_compressed(OUT/f"l26-motion-{name}.npz",positions=positions,
                            times_fs=np.arange(len(positions))*2,cell=atoms.cell.array)
        print(name,report["runs"][name],flush=True)
        (OUT/"l26-motion-provenance.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__":run()

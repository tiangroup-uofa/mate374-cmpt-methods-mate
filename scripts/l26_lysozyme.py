"""L26: hen egg-white lysozyme (PDB 1AKI, hydrogens added) drawn as atoms plus backbone trace (300 dpi).
Atom positions come from /tmp/l26fix/lys.npz, made with PDBFixer + OpenMM amber14/protein.ff14SB."""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
d=np.load(sys.argv[1] if len(sys.argv)>1 else "/tmp/l26fix/lys.npz")
pos=d["pos"]-d["pos"].mean(0); el=d["el"]; ca=d["ca"]
# principal axes for a clear view
w,v=np.linalg.eigh(np.cov(pos.T)); p=pos@v[:,::-1]
col={"C":"#8a8f98","N":"#1f77b4","O":"#d62728","S":"#e6b800","H":"#e8e8e8"}
size={"C":34,"N":34,"O":34,"S":60,"H":12}
order=np.argsort(p[:,2])
fig,ax=plt.subplots(figsize=(6.4,5.6))
ax.scatter(p[order,0],p[order,1],s=[size[e] for e in el[order]],c=[col[e] for e in el[order]],edgecolors="#5e5e5e",linewidths=0.2,alpha=0.85)
t=p[ca]; ax.plot(t[:,0],t[:,1],c="#231f20",lw=1.0,alpha=0.6)
for e,lab in (("C","carbon"),("N","nitrogen"),("O","oxygen"),("S","sulfur"),("H","hydrogen")):
    ax.scatter([],[],s=40,c=col[e],edgecolors="#5e5e5e",linewidths=0.3,label=lab)
ax.plot([],[],c="#231f20",lw=1,label="backbone")
ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.01),fontsize=10,ncol=6,handletextpad=0.2,columnspacing=0.9)
ax.set_aspect("equal"); ax.axis("off")
fig.savefig(ROOT/"assets/L26-lysozyme.png",dpi=300,bbox_inches="tight")

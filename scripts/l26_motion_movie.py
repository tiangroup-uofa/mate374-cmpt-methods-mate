"""Atomic-motion comparison from actual LJ/EAM/MACE trajectories (one (001) plane)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12})
names=["LJ","EAM","MACE"]
STEEL="#7a8a99"; BURG="#8c2d3c"; colours=[STEEL]*3
MODEL={"LJ":"#ff7f0e","EAM":"#1f77b4","MACE":"#2ca02c"}   # colours only for the displacement curves
line_styles={n:(c,"-") for n,c in MODEL.items()}
data=[np.load(ROOT/f"data/l26/l26-motion-{n}.npz") for n in names]
pos=[d["positions"] for d in data]; times=data[0]["times_fs"]
assert all(np.array_equal(pos[0][0],p[0]) for p in pos)
L=float(data[0]["cell"][0,0])
layer=np.where(np.isclose(pos[0][0][:,2],0,atol=1e-6))[0]      # 8 atoms in the z = 0 plane
SHIFT=L/8   # put lattice sites inside the drawn cell, not on its edges
start=np.mod(pos[0][0][layer,:2]+SHIFT,L)
centre=np.array([L/2,L/2]); hl=layer[np.argmin(np.linalg.norm(start-centre-0.9,axis=1))]
MAG=12
disp=[p[:,layer,:2]-p[0][layer,:2] for p in pos]
hl_i=int(np.where(layer==hl)[0][0])
fig=plt.figure(figsize=(8.6,5.8))
gs=fig.add_gridspec(2,3,height_ratios=[1.35,1],hspace=.42,wspace=.08,left=.15,right=.98,top=.93,bottom=.1)
spheres=[];trails=[]
for j,(name,c) in enumerate(zip(names,colours)):
    ax=fig.add_subplot(gs[0,j])
    ax.add_patch(Rectangle((0,0),L,L,fill=False,ec="#8a8f98",lw=1,ls="--"))
    for x,y in start: ax.add_patch(Circle((x,y),0.62,fc="none",ec="#b3b3b3",lw=1.2,zorder=1))
    sp=[ax.add_patch(Circle((x,y),0.55,fc=c,ec="white",lw=1,alpha=.9,zorder=3)) for x,y in start]
    sp[hl_i].set_facecolor(BURG); sp[hl_i].set_alpha(1); sp[hl_i].set_edgecolor("#231f20")
    spheres.append(sp)
    tr,=ax.plot([],[],c=BURG,lw=2,alpha=.6,zorder=4); trails.append(tr)
    ax.set(xlim=(-.15,L+.15),ylim=(-.15,L+.15),aspect="equal",xticks=[],yticks=[])
    ax.set_title(name,color="#231f20",fontweight="bold")
    for s in ax.spines.values(): s.set_visible(False)
axt=fig.add_subplot(gs[1,:])
for name,d in zip(names,disp):
    c,ls=line_styles[name]; axt.plot(times,d[:,hl_i,0],c=c,ls=ls,lw=2,label=name)
axt.axhline(0,c="#b3b3b3",lw=.8)
cursor=axt.axvline(0,c="#231f20",lw=1)
axt.set(xlabel="time (fs)",ylabel="centre atom\nx-displacement (Å)",xlim=(0,times[-1]))
axt.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.5,1.0))
axt.spines[["top","right"]].set_visible(False)
def draw(k):
    for sp,tr,d in zip(spheres,trails,disp):
        shown=start+MAG*d[k]
        for s,(x,y) in zip(sp,shown): s.center=(x,y)
        h=start[hl_i]+MAG*d[max(0,k-20):k+1,hl_i]
        tr.set_data(h[:,0],h[:,1])
    cursor.set_xdata([times[k]])
    return [s for sp in spheres for s in sp]+trails+[cursor]
seq=[0]*8+list(range(0,len(times),2))+[len(times)-1]*8
FuncAnimation(fig,draw,frames=seq).save(ROOT/"assets/L26-atomic-motion.gif",writer=PillowWriter(fps=12),dpi=90)
draw(len(times)-1); fig.savefig(ROOT/"assets/L26-atomic-motion-final.png",dpi=300)
draw(200); fig.savefig("/tmp/l26-motion-mid.png",dpi=120)
print("highlight",hl,"max shown offset Å",max(MAG*np.linalg.norm(d,axis=-1).max() for d in disp))

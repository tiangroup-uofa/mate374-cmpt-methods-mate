"""L26: two-atom LJ pair (L24 engine) pulled apart by a slowly ramped force; velocity Verlet, reduced units."""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12})
r0=2**(1/6); ri=(26/7)**(1/6); Fmax=24*(2*ri**-13-ri**-7)
dU=lambda r:24*(r**-7-2*r**-13)
# x1=-r/2, x2=r/2, m=1; equal and opposite pull F(t) on each atom
dt=0.002; ramp=2.6/60.0
x=np.array([-r0/2,r0/2]); v=np.zeros(2); t=0.0; rows=[]
def acc(x,t):
    r=x[1]-x[0]; f=-dU(r); F=ramp*t
    return np.array([-f-F, f+F])
a=acc(x,t)
while x[1]-x[0]<2.4:
    if round(t/dt)%100==0: rows.append((t,x[0],x[1],ramp*t))
    v+=0.5*dt*a; x+=dt*v; t+=dt; a=acc(x,t); v+=0.5*dt*a
rows=np.array(rows); r=rows[:,2]-rows[:,1]
k_break=np.argmax(r>ri+0.02); print("F at break",rows[k_break,3],"F_max",Fmax,"r",r[k_break])
fig,(ax,bx)=plt.subplots(2,1,figsize=(7,6.3),gridspec_kw={"height_ratios":[1,1.7],"hspace":.45})
fig.subplots_adjust(bottom=.14)
ax.set(xlim=(-1.7,1.7),ylim=(-.6,.6),aspect="equal",xticks=[],yticks=[]); [s.set_visible(False) for s in ax.spines.values()]
c1=ax.add_patch(Circle((0,0),.32,fc="#9cc3e4",ec="#231f20")); c2=ax.add_patch(Circle((0,0),.32,fc="#9cc3e4",ec="#231f20"))
bond,=ax.plot([],[],c="#5e5e5e",lw=2,zorder=0)
ar1=ax.annotate("",(0,0),(0,0),arrowprops=dict(arrowstyle="-|>",color="#ff7f0e",lw=2))
ar2=ax.annotate("",(0,0),(0,0),arrowprops=dict(arrowstyle="-|>",color="#ff7f0e",lw=2))
rr=np.linspace(1.0,2.4,600); bx.plot(rr,dU(rr),c="#1f77b4",lw=2.2,label="restoring force U′(r)")
bx.axvline(r0,c="#2ca02c",ls="--",lw=1); bx.axvline(ri,c="#b5473a",ls="--",lw=1)
bx.axhline(0,c="#8a8f98",lw=.8,ls=":")
fl=bx.axhline(0,c="#ff7f0e",lw=1.6,label="applied pull F(t)"); dot,=bx.plot([],[],"o",c="#231f20",ms=8)
bx.set(xlim=(1.0,2.4),ylim=(-3,3),xlabel="separation r / σ",ylabel="force (ε/σ)")
bx.set_xticks([1.0,1.5,2.0,r0,ri]); bx.set_xticklabels(["1","1.5","2","\nr$_0$≈1.12","\nr$_{infl}$≈1.24"])
for tl,c,h in zip(bx.get_xticklabels()[-2:],("#2ca02c","#b5473a"),("right","left")): tl.set_color(c); tl.set_ha(h)
bx.legend(frameon=False,loc="lower right",fontsize=10); bx.spines[["top","right"]].set_visible(False)
title=fig.text(.5,.94,"",ha="center",fontsize=13)
def draw(k):
    _,x1,x2,F=rows[k]; R=x2-x1
    c1.center=(x1,0); c2.center=(x2,0)
    bond.set_data([x1,x2],[0,0]) if R<ri+0.15 else bond.set_data([],[])
    L=0.1+0.25*F/abs(Fmax)
    ar1.xy=(x1-.34-L,0); ar1.xyann=(x1-.34,0); ar2.xy=(x2+.34+L,0); ar2.xyann=(x2+.34,0)
    fl.set_ydata([F]); dot.set_data([R],[dU(R)])
    title.set_text("bond holds: U′(r) balances the pull" if R<ri else "past r$_{infl}$: no larger restoring force, the bond breaks")
    title.set_color("#231f20" if R<ri else "#b5473a")
seq=[0]*15+list(range(0,k_break,max(1,k_break//90)))+list(range(k_break,len(rows)))+[len(rows)-1]*20
FuncAnimation(fig,draw,frames=seq).save(ROOT/"assets/L26-bond-pull.gif",writer=PillowWriter(fps=15),dpi=150)
draw(len(rows)-1); fig.savefig(ROOT/"assets/L26-bond-pull-final.png",dpi=300)

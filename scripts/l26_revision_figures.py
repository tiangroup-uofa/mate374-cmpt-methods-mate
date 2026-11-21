"""Checked L26 figures. All raster outputs are saved at 300 dpi."""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import PchipInterpolator
from ase.build import bulk
from l26_models import ROOT, EPS, SIG, calculator, minimum
from l26_figures import read_setfl, eam_functions

ASSETS=ROOT/"assets"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
BLUE,RED,GREEN="#1f5fa8","#bd3e34","#27845a"
def save(fig,name):
    fig.savefig(ASSETS/f"{name}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

def pair():
    r=np.linspace(.97,2.4,1000); r0=2**(1/6); ri=(26/7)**(1/6)
    u=4*(r**-12-r**-6)
    up=24*(r**-7-2*r**-13)
    upp=24*(26*r**-14-7*r**-8)
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),layout="constrained")
    for ax,y,title,yl,lim in zip(axes,[u,up,upp],
            ["Energy","First derivative","Second derivative"],
            [r"$U/\varepsilon$",r"$U'\sigma/\varepsilon$",r"$U''\sigma^2/\varepsilon$"],
            [(-1.2,2),(-4,3),(-15,85)]):
        ax.plot(r,y,color=BLUE); ax.axhline(0,color=".6",lw=.6)
        ax.axvline(r0,color=GREEN,ls=":"); ax.axvline(ri,color=RED,ls=":")
        ax.set(xlim=(.97,2.4),ylim=lim,xlabel=r"$r/\sigma$",ylabel=yl,title=title)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].annotate(r"$r_0,\ U=-\varepsilon$",(r0,-1),(1.45,-.7),arrowprops={"arrowstyle":"->"})
    axes[1].annotate(r"maximum $U'=|F_r|$",(ri,2.396),(1.43,1.4),arrowprops={"arrowstyle":"->"})
    axes[2].annotate(r"$k=57.15\,\varepsilon/\sigma^2$",(r0,57.146),(1.4,60),arrowprops={"arrowstyle":"->"})
    axes[2].annotate("inflection of U",(ri,0),(1.5,20),arrowprops={"arrowstyle":"->"})
    save(fig,"L26-pair-derivatives")
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
    axes[0].axvspan(.65,1,color=".88")
    axes[0].plot([1,2.5],[0,0],color=BLUE,lw=2)
    axes[0].annotate("",(1,2.6),(1,0),arrowprops={"arrowstyle":"->","lw":2,"color":BLUE})
    axes[0].text(.68,1.3,"forbidden\noverlap",fontsize=10)
    axes[0].text(1.06,2.1,r"$U\to\infty$")
    axes[0].set(title="Hard sphere",xlabel="r / d",ylabel="energy (arbitrary scale)",xlim=(.65,2.5),ylim=(-1.2,2.7))
    axes[1].plot(r,u,color=RED,lw=2); axes[1].axhline(0,color=".6",lw=.6)
    axes[1].set(title="Lennard–Jones",xlabel=r"$r/\sigma$",ylabel=r"$U/\varepsilon$",ylim=(-1.2,2.7))
    save(fig,"L26-hard-sphere-potential")

def cells():
    from itertools import product
    fig=plt.figure(figsize=(12,3.4))
    for i,(s,z) in enumerate(zip(["diamond","sc","bcc","fcc"],[4,6,8,12])):
        ax=fig.add_subplot(1,4,i+1,projection="3d")
        atoms=bulk("Cu",s,a=1,cubic=True)
        points=atoms.get_scaled_positions()
        tiled=np.vstack([points+v for v in product([0,1],repeat=3)])
        tiled=tiled[np.all(tiled<=1.00001,axis=1)]
        ax.scatter(*tiled.T,s=45,color=BLUE,depthshade=True)
        for p in product([0,1],repeat=3):
            for d in range(3):
                if p[d]==0:
                    q=list(p);q[d]=1
                    ax.plot(*np.array([p,q]).T,color=".65",lw=.8)
        ax.set_title(f"{s.upper() if s!='diamond' else 'Diamond'} · Z = {z}")
        ax.set_axis_off();ax.set_box_aspect((1,1,1));ax.view_init(22,28)
    save(fig,"L26-crystal-cells")

def cohesion():
    rho,emb,r,f,rphi,rc=read_setfl(ROOT/"activities/Cu_mishin1.eam.alloy")
    d=3.615/2**.5; z=np.linspace(0,12,301)
    from scipy.interpolate import InterpolatedUnivariateSpline
    embfun=InterpolatedUnivariateSpline(rho,emb)
    phi=np.interp(d,r[1:],rphi[1:]/r[1:])
    dens=np.interp(d,r,f)
    pair=4*EPS*((SIG/d)**12-(SIG/d)**6)
    # Positive binding relative to isolated atom: retain F(0) reference.
    curves=[-.5*z*pair,-.5*z*phi-embfun(z*dens)+embfun(0)]
    structures=["diamond","sc","bcc","fcc"]; zs=np.array([4,6,8,12])
    refs=np.array([2.293,2.996,3.496,3.540])
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
    for y,c,label in zip(curves,[RED,BLUE],["LJ, nearest neighbours","EAM, nearest neighbours"]):
        axes[0].plot(z,y,c=c,label=label)
    for name,c in [("LJ",RED),("EAM",BLUE)]:
        values=np.array([-minimum(name,s).fun for s in structures])
        xx=np.linspace(4,12,200)
        axes[1].plot(xx,PchipInterpolator(zs,values)(xx),c=c,label=name)
        axes[1].scatter(zs,values,c=c,s=35)
    for ax in axes:
        for x,y,s,m in zip(zs,refs,structures,["D","s","^","o"]):
            ax.scatter([x],[y],marker=m,c="black",s=42,zorder=5)
            ax.annotate(s.upper() if s!="diamond" else "diamond",(x,y),xytext=(0,8),textcoords="offset points",ha="center",fontsize=9)
        ax.scatter([],[],c="black",label="Copper: quantum chemistry")
        ax.set(xlabel="nearest-neighbour coordination Z",ylabel="cohesive energy (eV/atom)",ylim=(0,4.15),xlim=(-.2,13.4))
        ax.legend(loc="lower right",fontsize=9,frameon=False)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_title("Fixed spacing, nearest neighbours")
    axes[1].set_title("Full crystals, optimized spacing")
    save(fig,"L26-cohesion-checked")

def nve():
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for name,c in [("LJ",RED),("EAM",BLUE),("MACE",GREEN)]:
        for dt,ls in [(1,"--"),(.5,"-")]:
            data=np.genfromtxt(ROOT/f"data/l26/l26-nve-{name}-{dt:g}fs.csv",delimiter=",",names=True)
            t=data["time_fs"];u=data["potential_eV_per_atom"];k=data["kinetic_eV_per_atom"]
            if dt==.5:axes[0].plot(t,1000*(u-u[0]),c=c,label=name)
            axes[1].plot(t,1e6*(u+k-u[0]-k[0]),c=c,ls=ls,label=f"{name}, {dt:g} fs")
    axes[0].set(xlabel="time (fs)",ylabel="potential energy change (meV/atom)",title="Energy exchange, 0.5 fs step")
    axes[1].set(xlabel="time (fs)",ylabel="total-energy error (µeV/atom)",title="Time-step convergence")
    for ax in axes:ax.legend(frameon=False,fontsize=9);ax.spines[["top","right"]].set_visible(False)
    save(fig,"L26-small-crystal-nve")

def hard_frames(pos,vel,L,diameter,times):
    # Reflecting walls at particle-centre bounds 0,L; exact pair and wall events.
    pos=pos.copy();vel=vel.copy();t=0.;frames=[]; iu=np.triu_indices(len(pos),1)
    for target in times:
        while t<target-1e-12:
            dr=pos[iu[1]]-pos[iu[0]];dv=vel[iu[1]]-vel[iu[0]]
            b=np.sum(dr*dv,axis=1);vv=np.sum(dv*dv,axis=1)
            disc=b*b-vv*(np.sum(dr*dr,axis=1)-diameter**2)
            tc=np.full(len(b),np.inf);ok=(b<0)&(disc>0)&(vv>0)
            tc[ok]=np.maximum(0,(-b[ok]-np.sqrt(disc[ok]))/vv[ok])
            wall=np.where(vel>0,(L-pos)/np.where(vel!=0,vel,1),-pos/np.where(vel!=0,vel,1))
            wall[vel==0]=np.inf
            k=np.argmin(tc);w=np.unravel_index(np.argmin(wall),wall.shape)
            dt=min(tc[k],wall[w],target-t)
            pos+=vel*dt;t+=dt
            if dt==tc[k]:
                i,j=iu[0][k],iu[1][k];n=(pos[j]-pos[i])/diameter
                v=np.dot(vel[j]-vel[i],n);vel[i]+=v*n;vel[j]-=v*n
            elif dt==wall[w]:vel[w]*=-1
        frames.append(pos.copy())
    return np.array(frames)

def soft_frames(pos,vel,L,temp,times):
    pos=pos.copy();vel=vel.copy();rng=np.random.default_rng(374);dt=.004
    def force(x):
        dr=x[:,None]-x[None,:];r2=np.sum(dr*dr,axis=-1);np.fill_diagonal(r2,np.inf)
        q=np.where(r2<2.5**2,r2**-3,0)
        return np.sum((24*q*(2*q-1)/r2)[:,:,None]*dr,axis=1)
    def drift(x,v):
        x=x+.5*dt*v
        hits=(x<0)|(x>L)
        v=np.where(hits,-v,v);x=np.mod(x,2*L);x=np.where(x>L,2*L-x,x)
        return x,v
    f=force(pos);frames=[];step=0;c=np.exp(-.5*dt)
    for target in times:
        while step<round(target/dt):
            vel+=.5*dt*f;pos,vel=drift(pos,vel)
            vel=c*vel+np.sqrt((1-c*c)*temp)*rng.normal(size=vel.shape)
            pos,vel=drift(pos,vel);f=force(pos);vel+=.5*dt*f;step+=1
        frames.append(pos.copy())
    return np.array(frames)

def movie():
    L=12.;m=6
    grid=(np.stack(np.meshgrid(np.arange(m),np.arange(m)),-1).reshape(-1,2)+.5)*L/m
    times=np.linspace(0,60,81); data=[]
    for temp in (.15,.8):
        v=np.random.default_rng(374).normal(0,np.sqrt(temp),grid.shape);v-=v.mean(0)
        data.extend([hard_frames(grid,v,L,1,times),soft_frames(grid,v,L,temp,times)])
    fig,axes=plt.subplots(2,2,figsize=(6.8,6.8))
    fig.subplots_adjust(bottom=.08,hspace=.25,wspace=.1)
    artists=[]
    labels=["Hard particles · scaled temperature 0.15","Lennard–Jones · condensed phase","Hard particles · scaled temperature 0.8","Lennard–Jones · gas phase"]
    for ax,title,c in zip(axes.flat,labels,[RED,BLUE]*2):
        ax.set(xlim=(-.5,L+.5),ylim=(-.5,L+.5),aspect="equal",xticks=[],yticks=[])
        ax.set_title(title,fontsize=10)
        artists.append(ax.scatter(grid[:,0],grid[:,1],s=65,c=c,edgecolor="white",linewidth=.4))
    label=fig.text(.5,.02,"",ha="center",fontsize=9)
    def draw(k):
        for a,frames in zip(artists,data):a.set_offsets(frames[k])
        label.set_text(f"Scaled time t = {times[k]:.1f}")
        return artists+[label]
    sequence=[0]*36+list(range(len(times)))+[len(times)-1]*12
    FuncAnimation(fig,draw,frames=sequence).save(ASSETS/"L26-lattice-comparison.gif",writer=PillowWriter(fps=12),dpi=300)
    draw(len(times)-1);save(fig,"L26-lattice-comparison-final")
    np.savez_compressed(ROOT/"data/l26/l26-lattice-animation.npz",times=times,positions=np.array(data))

if __name__=="__main__":
    import sys
    jobs={"pair":pair,"cells":cells,"cohesion":cohesion,"eam":eam_functions,"nve":nve,"movie":movie}
    for job in sys.argv[1:]: jobs[job]()

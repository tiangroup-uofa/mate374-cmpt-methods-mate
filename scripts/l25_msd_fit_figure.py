"""L25 atomic motion and one-species MSD fit from the supplied Atomify trajectory.

Run: uv run --locked python scripts/l25_msd_fit_figure.py
Input: data/l25/atomify-diffusion.lammpstrj.gz
The supplied input uses dt=0.005 tau. Only five selected light atoms enter the display and fit.
"""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/l25"
ASSETS=ROOT/"assets"
DPI=300

def load_trajectory():
    steps,positions=[],[]
    initial_types=None
    initial_bounds=None
    with gzip.open(DATA/"atomify-diffusion.lammpstrj.gz","rt") as handle:
        while True:
            line=handle.readline()
            if not line:
                break
            assert line.strip()=="ITEM: TIMESTEP"
            steps.append(int(handle.readline()))
            assert handle.readline().strip()=="ITEM: NUMBER OF ATOMS"
            n=int(handle.readline())
            assert n==6750
            assert handle.readline().startswith("ITEM: BOX BOUNDS")
            bounds=np.array([[float(x) for x in handle.readline().split()] for _ in range(3)])
            assert handle.readline().strip()=="ITEM: ATOMS id type xu yu zu"
            atoms=np.loadtxt([handle.readline() for _ in range(n)])
            atoms=atoms[np.argsort(atoms[:,0])]
            assert np.array_equal(atoms[:,0],np.arange(1,n+1))
            types=atoms[:,1].astype(int)
            if initial_types is None:
                initial_types=types
                initial_bounds=bounds
            else:
                assert np.array_equal(types,initial_types)
                assert np.array_equal(bounds,initial_bounds)
            positions.append(atoms[:,2:])
    assert np.array_equal(steps,np.arange(0,10001,100))
    positions=np.asarray(positions)
    assert np.isfinite(positions).all()
    return np.asarray(steps),positions,initial_types,initial_bounds

def analyse(steps,positions,chosen):
    times=steps*.005
    selected=positions[:,chosen]
    displacement=selected-selected[0]
    squared=np.sum(displacement**2,axis=-1)
    msd=squared.mean(axis=1)
    assert len(chosen)==5 and msd[0]==0
    fit=times>=10
    slope,intercept=np.polyfit(times[fit],msd[fit],1)
    header="time_tau,msd_light_sigma2,"+",".join(
        f"r2_atom{atom+1}_sigma2" for atom in chosen)
    np.savetxt(DATA/"atomify-diffusion-msd.csv",
               np.column_stack([times,msd,squared]),delimiter=",",
               header=header,comments="",fmt="%.12g")
    source_hash=hashlib.sha256(gzip.decompress(
        (DATA/"atomify-diffusion.lammpstrj.gz").read_bytes())).hexdigest()
    report={
        "source":"instructor-supplied Atomify/LAMMPS trajectory",
        "source_sha256_uncompressed":source_hash,
        "selected_atom_ids":(chosen+1).tolist(),
        "selection":"five type-1 atoms, random seed 374, without replacement",
        "frames":len(times),"dt_tau":.005,"sample_steps":100,
        "MSD":"mean of squared 3D unwrapped displacements of these same five atoms",
        "fit_time_tau":[10,50],
        "slope_sigma2_per_tau":float(slope),
        "intercept_sigma2":float(intercept),
        "D_sigma2_per_tau":float(slope/6),
        "final_MSD_sigma2":float(msd[-1]),
    }
    (DATA/"atomify-diffusion-fit-results.txt").write_text(
        "\n".join(f"{key}: {json.dumps(value)}" for key,value in report.items())+"\n")
    print(json.dumps(report,indent=2),flush=True)
    return times,msd,slope,intercept

def render(selected,times,msd,slope,intercept):
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans"],
        "font.size":11,"axes.labelsize":11,"xtick.labelsize":9,"ytick.labelsize":10,
        "axes.linewidth":.8,
    })
    fig=plt.figure(figsize=(7.4,3.55),dpi=DPI)
    gs=fig.add_gridspec(1,2,width_ratios=[1.1,1],left=.015,right=.985,
                       bottom=.18,top=.88,wspace=.25)
    atoms_axis=fig.add_subplot(gs[0,0],projection="3d")
    graph=fig.add_subplot(gs[0,1])
    colours=["#1f77b4","#b5473a","#2ca02c","#ff7f0e","#9467bd"]
    trails=[]
    dots=[]
    for j,col in enumerate(colours):
        trail,=atoms_axis.plot([],[],[],c=col,lw=1.15)
        dot=atoms_axis.scatter(*selected[0,j],s=30,c=col,
                              edgecolors="#231f20",linewidths=.35,depthshade=False)
        atoms_axis.scatter(*selected[0,j],s=13,c=col,marker="x",linewidths=.8)
        trails.append(trail)
        dots.append(dot)
    lower=selected.min(axis=(0,1))
    upper=selected.max(axis=(0,1))
    padding=(upper-lower)*.08
    atoms_axis.set(xlim=(lower[0]-padding[0],upper[0]+padding[0]),
                   ylim=(lower[1]-padding[1],upper[1]+padding[1]),
                   zlim=(lower[2]-padding[2],upper[2]+padding[2]),
                   xlabel="x (σ)",ylabel="y (σ)",zlabel="z (σ)")
    atoms_axis.set_box_aspect(upper-lower+2*padding)
    atoms_axis.view_init(elev=24,azim=-65)
    from matplotlib.ticker import MaxNLocator
    for axis in [atoms_axis.xaxis,atoms_axis.yaxis,atoms_axis.zaxis]:
        axis.set_major_locator(MaxNLocator(3))
        axis.pane.fill=False
    atoms_axis.grid(False)
    atoms_axis.set_title("Five atom trajectories",fontsize=12,pad=8)
    clock=atoms_axis.text2D(.5,-.1,"",ha="center",transform=atoms_axis.transAxes,fontsize=11)

    fit=times>=10
    graph.plot(times,msd,".",color="#1f77b4",ms=3.5,label="Mean of 5 atoms")
    graph.plot(times[fit],slope*times[fit]+intercept,
               color="#231f20",lw=1.5,label="Linear fit")
    cursor=graph.axvline(0,color="#8a8f98",lw=.7,ls=":")
    current,=graph.plot([],[],"o",c="#1f77b4",ms=4)
    graph.set(xlabel="Time t (τ)",ylabel="MSD (σ²)",
              title="Five-atom MSD",xlim=(0,50),ylim=(0,float(msd.max())*1.14))
    graph.spines[["top","right"]].set_visible(False)
    graph.legend(frameon=False,loc="upper left",fontsize=10,handlelength=1.4)
    graph.text(.96,.08,f"s = {slope:.3f} σ²/τ\nD = s/6 = {slope/6:.4f} σ²/τ",
               transform=graph.transAxes,ha="right",va="bottom",fontsize=10.5)

    def draw(k):
        for j,(trail,dot) in enumerate(zip(trails,dots)):
            trail.set_data_3d(*selected[:k+1,j].T)
            dot._offsets3d=tuple(selected[k:k+1,j].T)
        clock.set_text(f"t = {times[k]:.1f} τ")
        cursor.set_xdata([times[k],times[k]])
        current.set_data([times[k]],[msd[k]])

    frames=[]
    palette=None
    sequence=[0]*5+list(range(len(times)))+[len(times)-1]*10
    for k in sequence:
        draw(k)
        fig.canvas.draw()
        im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3])
        if palette is None:
            palette=im.quantize(colors=128)
        frames.append(im.quantize(palette=palette,dither=Image.Dither.NONE))
    frames[0].save(ASSETS/"L25-atomify-diffusion.gif",
                   save_all=True,append_images=frames[1:],duration=90,
                   loop=0,optimize=True,disposal=1,dpi=(DPI,DPI))
    draw(len(times)-1)
    fig.savefig(ASSETS/"L25-analysis-msd.png",dpi=DPI)
    plt.close(fig)
    print("Saved five-atom GIF and static poster at 300 dpi.",flush=True)

if __name__=="__main__":
    steps,positions,types,bounds=load_trajectory()
    chosen=np.random.default_rng(374).choice(np.flatnonzero(types==1),5,replace=False)
    times,msd,slope,intercept=analyse(steps,positions,chosen)
    render(positions[:,chosen],times,msd,slope,intercept)

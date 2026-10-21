"""ASE plot_atoms storyboard and animation of an actual MD trajectory."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from ase.io import read
from ase.visualize.plot import plot_atoms
from ase.utils import rotate
from analyze_sampling import analyze, plot_summary, pca_axes, histogram_axes


def aligned_frames(frames):
    reference=frames[0].positions[::3].copy()
    reference-=reference.mean(0)
    aligned=[]
    for frame in frames:
        a=frame.copy(); p=a.positions[::3]; center=p.mean(0)
        u,_,vh=np.linalg.svd((p-center).T@reference)
        fix=np.eye(3); fix[-1,-1]=np.linalg.det(u@vh)
        a.positions=(a.positions-center)@u@fix@vh
        aligned.append(a)
    # Choose one fixed view that separates oxygen spheres in key frames.
    keys=sorted(set(np.linspace(0,len(aligned)-1,6).astype(int).tolist()+[len(aligned)//2]))
    vectors=[]
    for idx in keys:
        r=aligned[idx].positions[::3]
        vectors.extend([r[i]-r[j] for i in range(6) for j in range(i)])
    vectors=np.array(vectors)
    best=-1.0; camera=np.eye(3)
    for x in range(0,180,15):
        for y in range(0,180,15):
            candidate=rotate(f'{x}x,{y}y,-15z')
            separation=np.linalg.norm((vectors@candidate)[:,:2],axis=1).min()
            if separation>best: best=separation;camera=candidate
    for a in aligned: a.positions=a.positions@camera
    return aligned


def draw_atoms(ax, atoms):
    ax.clear()
    p=atoms.positions
    # Fixed ASE bounding box; plot_atoms shifts its lower corner to (0,0).
    for i in range(0,18,3):
        for j in (i+1,i+2):
            ax.plot(p[[i,j],0]+4,p[[i,j],1]+4,color='#555555',lw=2,zorder=0)
    hb=0
    for o in range(0,18,3):
        for h in (o+1,o+2):
            for acceptor in range(0,18,3):
                if acceptor==o: continue
                oh=p[h]-p[o]; ha=p[acceptor]-p[h]
                if np.linalg.norm(ha)<2.4 and np.dot(oh,ha)/(np.linalg.norm(oh)*np.linalg.norm(ha))>np.cos(np.deg2rad(30)):
                    ax.plot(p[[h,acceptor],0]+4,p[[h,acceptor],1]+4,color='#5894a0',lw=1,ls='--',zorder=0)
                    hb+=1
    plot_atoms(atoms,ax=ax,show_unit_cell=0,bbox=(-4,-4,4,4),radii=.55,
               colors=['#db413d' if z==8 else '#f0f0f0' for z in atoms.numbers])
    ax.set_xlim(0,8);ax.set_ylim(0,8);ax.set_aspect('equal');ax.axis('off')
    return hb


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path,nargs='?',default=Path('cv-scan'))
    args=parser.parse_args();out=args.directory
    frames=read(out/'trajectory.traj',':');aligned=aligned_frames(frames)
    sampling=analyze(out)
    plot_summary(out,sampling)
    structural=np.column_stack([sampling['time']*1000,sampling['rmsd']])
    data=np.loadtxt(out/'observables.csv',delimiter=',',skiprows=1)
    names=sampling['names']
    gallery=list(sampling['refs'].values())
    fig,axes=plt.subplots(3,3,figsize=(10,10),layout='constrained')
    for ax,name,a in zip(axes.flat,names,gallery):
        draw_atoms(ax,aligned_frames([a])[0]);ax.set_title(name)
    fig.suptitle('Source conformations · before TIP3P relaxation')
    fig.savefig(out/'reference-gallery.png',dpi=150);plt.close(fig)
    chosen=np.linspace(0,len(frames)-1,6).astype(int)
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for ax,idx in zip(axes.flat,chosen):
        hb=draw_atoms(ax,aligned[idx]);f=frames[idx]
        distances=structural[idx,1:];nearest=names[np.argmin(distances)]
        ax.set_title(f'{f.info["time_fs"]/1000:.2f} ps  |  Rg {f.info["rg"]:.2f} Å\n'
                     f'nearest source O geometry: {nearest} ({min(distances):.2f} Å RMSD)',fontsize=10)
    fig.suptitle('Book → prism-target → book-target CV scan\nASE plot_atoms • rigid TIP3P • dashed lines: geometric H bonds',fontsize=14)
    fig.savefig(out/'ase-storyboard.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8.5),layout='constrained')
    ax,cvax,pcaax,histax=axes.flat
    pca_axes(pcaax,sampling)
    mesh=histogram_axes(histax,sampling)
    fig.colorbar(mesh,ax=histax,label='Saved-frame fraction')
    pc_marker,=pcaax.plot([],[],'o',color='#00bfff',mec='black',ms=8,zorder=5)
    cv_marker,=histax.plot([],[],'o',color='#00bfff',mec='black',ms=8,zorder=5)
    cvax.plot(data[:,0]/1000,data[:,1],color='#ce473b',label='Rg')
    cvax.plot(data[:,0]/1000,data[:,3],'k--',lw=1,label='moving target')
    cvax.set(xlabel='Time (ps)',ylabel='Oxygen Rg (Å)');cvax.legend(frameon=False)
    marker=cvax.axvline(0,color='#468a9c',lw=1)
    title=fig.suptitle('')
    def draw(idx):
        draw_atoms(ax,aligned[idx]);f=frames[idx];t=f.info['time_fs']/1000
        marker.set_xdata([t,t]);nearest=names[np.argmin(structural[idx,1:])]
        pc_marker.set_data([sampling['points'][idx,0]],[sampling['points'][idx,1]])
        cv_marker.set_data([sampling['cv'][idx,0]],[sampling['cv'][idx,1]])
        title.set_text(f'ASE plot_atoms • {t:.2f} ps • nearest source O geometry: {nearest}')
    indices=list(range(0,len(frames),max(1,len(frames)//100)))
    if indices[-1]!=len(frames)-1:indices.append(len(frames)-1)
    animation=FuncAnimation(fig,draw,frames=indices,interval=100)
    animation.save(out/'ase-trajectory.gif',writer=PillowWriter(fps=10),dpi=110)
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True,layout='constrained')
    for ax,actual,target,label in zip(axes[:2],(1,2),(3,4),('Oxygen Rg (Å)','Mean O coordination')):
        ax.plot(data[:,0]/1000,data[:,actual],label='measured')
        ax.plot(data[:,0]/1000,data[:,target],'k--',label='target')
        ax.set_ylabel(label);ax.legend(frameon=False)
    for i,name in enumerate(names):axes[2].plot(structural[:,0]/1000,structural[:,i+1],label=name)
    axes[2].set(xlabel='Time (ps)',ylabel='Oxygen RMSD (Å)');axes[2].legend(ncol=3,frameon=False)
    fig.savefig(out/'scan-diagnostics.png',dpi=180);plt.close(fig)
    print('Saved reference gallery, ASE storyboard, four-panel GIF, sampling maps and diagnostics.')


if __name__=='__main__':main()

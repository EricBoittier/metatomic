"""Unbiased quenches to distinguish target matching from reaching an isomer basin."""
import argparse
import json
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.calculators.tip3p import TIP3P
from ase.optimize import BFGS
from scan_cv import oxygen_rmsd, PERMUTATIONS
from run_md import descriptors


def hb_graph(atoms):
    p=atoms.positions
    graph=np.zeros((6,6),dtype=int)
    for i in range(6):
        for h in (3*i+1,3*i+2):
            oh=p[h]-p[3*i]
            for j in range(6):
                if i==j:continue
                ha=p[3*j]-p[h]
                if np.linalg.norm(ha)<2.4 and np.dot(oh,ha)/(np.linalg.norm(oh)*np.linalg.norm(ha))>np.cos(np.deg2rad(30)):
                    graph[i,j]=1
    return graph


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path,nargs='?',default=Path('cv-scan'))
    out=parser.parse_args().directory
    summary=json.loads((out/'summary.json').read_text())
    frames=read(out/'trajectory.traj',':')
    names=list(summary['reference_CVs'])
    refs={n:read(out/f'{n}-reference.extxyz') for n in names}
    graphs={n:hb_graph(a) for n,a in refs.items()}
    results={}
    # End of forward hold and end of reverse hold.
    for label,step in [('forward',summary['stage_steps'][2]),('reverse',summary['stage_steps'][-1])]:
        frame=min(frames,key=lambda a:abs(a.info['time_fs']-.5*step)).copy()
        before={n:oxygen_rmsd(frame,a) for n,a in refs.items()}
        frame.calc=TIP3P(rc=20,width=1)
        converged=BFGS(frame,logfile=str(out/f'quench-{label}.log'),maxstep=.03).run(fmax=.01,steps=600)
        rmsd={n:oxygen_rmsd(frame,a) for n,a in refs.items()}
        g=hb_graph(frame)
        mismatch={n:int(min(np.abs(g[perm][:,perm]-graphs[n]).sum() for perm in PERMUTATIONS)) for n in names}
        results[label]=dict(time_fs=.5*step,converged=bool(converged),energy_eV=frame.get_potential_energy(),
                            CVs=descriptors(frame),oxygen_rmsd_before_A=before,oxygen_rmsd_quenched_A=rmsd,
                            hydrogen_bonds=int(g.sum()),directed_hbond_graph_mismatches=mismatch,
                            nearest_reference=min(rmsd,key=rmsd.get))
        write(out/f'{label}-quenched.extxyz',frame)
    (out/'endpoint-analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()

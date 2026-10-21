"""Bidirectional book/prism CV scan with pinned-core CVs and rigid TIP3P MD."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from ase.calculators.plumed import Plumed
from ase.constraints import FixBondLengths, FixCom
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import thermalize_momenta
from bias_utils import HERE, KERNEL, cv_input
from run_md import descriptors
from conformations import load_conformations

PAIRS = [(i+a,i+b) for i in range(0,18,3) for a,b in ((0,1),(0,2),(1,2))]
PERMUTATIONS = np.array(list(itertools.permutations(range(6))))


def prepare(atoms, log):
    atoms = atoms.copy()
    atoms.set_pbc(False)
    atoms.set_cell(np.zeros((3,3)))
    for i in range(0,18,3):
        u,v = atoms.positions[i+1:i+3]-atoms.positions[i]
        u /= np.linalg.norm(u); v /= np.linalg.norm(v)
        x=(u+v)/np.linalg.norm(u+v); y=(u-v)/np.linalg.norm(u-v)
        theta=np.deg2rad(angleHOH/2)
        atoms.positions[i+1]=atoms.positions[i]+rOH*(np.cos(theta)*x+np.sin(theta)*y)
        atoms.positions[i+2]=atoms.positions[i]+rOH*(np.cos(theta)*x-np.sin(theta)*y)
    atoms.set_constraint([FixBondLengths(PAIRS,tolerance=1e-12), FixCom()])
    atoms.calc=TIP3P(rc=20,width=1)
    if not BFGS(atoms,logfile=str(log),maxstep=.05).run(fmax=.01,steps=700):
        raise RuntimeError('Reference relaxation failed')
    return atoms


def oxygen_rmsd(atoms, reference):
    """Minimum over all 720 oxygen permutations and proper rotations (no reflections)."""
    p=atoms.positions[::3]; p=p-p.mean(0)
    q=reference.positions[::3]; q=q-q.mean(0)
    q=q[PERMUTATIONS]
    u,_,vh=np.linalg.svd(np.einsum('ia,pib->pab',p,q))
    fix=np.ones((len(q),3)); fix[:,-1]=np.linalg.det(u@vh)
    rot=(u*fix[:,None,:])@vh
    aligned=np.einsum('ia,pab->pib',p,rot)
    return float(np.sqrt(((aligned-q)**2).sum((1,2))/6).min())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE/'cv-scan')
    parser.add_argument('--leg-steps',type=int,default=1600)
    parser.add_argument('--hold-steps',type=int,default=400)
    args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(exist_ok=False,parents=True)
    raw=load_conformations()
    names=list(raw)
    refs={name:prepare(a,out/f'relax-{name}.log') for name,a in raw.items()}
    for name,a in refs.items(): write(out/f'{name}-reference.extxyz',a)
    endpoints={name:descriptors(a) for name,a in refs.items()}
    print('Relaxed reference CVs:',endpoints,flush=True)
    start=np.array(endpoints['book']); end=np.array(endpoints['prism'])
    leg,hold=args.leg_steps,args.hold_steps
    stages=[0,leg,leg+hold,2*leg+hold,2*(leg+hold)]
    targets=[start,end,end,start,start]
    strength=np.array([30.,5.]) # eV/A^2 for Rg, eV for dimensionless CN
    lines=cv_input()+['scan: MOVINGRESTRAINT ARG=rg,cn '+' '.join(
        f'STEP{i}={step} AT{i}={target[0]:.16g},{target[1]:.16g} KAPPA{i}=30,5'
        for i,(step,target) in enumerate(zip(stages,targets))),
        f'PRINT ARG=rg,cn,scan.bias,scan.rg_cntr,scan.cn_cntr,scan.work STRIDE=1 FILE={out/"COLVAR"} FMT=%.12f']
    (out/'plumed.dat').write_text('\n'.join(lines)+'\n')
    atoms=refs['book'].copy()
    thermalize_momenta(atoms,temperature_K=150,rng=np.random.default_rng(20261017))
    calc=Plumed(TIP3P(rc=20,width=1),input=lines,timestep=.5*units.fs,atoms=atoms,
                kT=150*units.kB,log=str(out/'plumed.log'))
    atoms.calc=calc
    lengths=np.array([atoms.get_distance(i,j) for i,j in PAIRS])
    rows=[]; bond_error=0.
    with Trajectory(str(out/'trajectory.traj'),'w',atoms) as traj:
        with Langevin(atoms,.5*units.fs,temperature_K=150,friction=.01/units.fs,
                      fixcm=False,rng=np.random.default_rng(1729)) as dyn:
            def record():
                nonlocal bond_error
                cv=np.array(descriptors(atoms)); step=dyn.nsteps
                target=np.array([np.interp(step,stages,[t[k] for t in targets]) for k in range(2)])
                bias=.5*np.sum(strength*(cv-target)**2)
                energy=atoms.get_potential_energy()
                rows.append([step*.5,*cv,*target,bias,energy-bias,atoms.get_temperature()])
                assert np.isfinite(rows[-1]).all()
                bond_error=max(bond_error,float(np.max(np.abs(np.array([atoms.get_distance(i,j) for i,j in PAIRS])-lengths))))
                assert bond_error<1e-8
                r=atoms.positions[::3]
                assert np.linalg.norm(r[:,None]-r[None,:],axis=2).max()<19
                if step%20==0 or step in stages:
                    atoms.info.update(time_fs=step*.5,rg=float(cv[0]),cn=float(cv[1]))
                    traj.write()
                if step%500==0: print('step',step,'CV',cv,'target',target,flush=True)
            dyn.attach(record,interval=1)
            dyn.run(stages[-1])
    calc.plumed.finalize()
    a=np.array(rows); c=np.loadtxt(out/'COLVAR')
    assert len(c)==len(a)==stages[-1]+1
    assert np.max(abs(c[:,0:3]-a[:,0:3]))<1e-9
    assert np.max(abs(c[:,3]-a[:,5]))<1e-8
    assert np.max(abs(c[:,4:6]-a[:,3:5]))<1e-9
    np.savetxt(out/'observables.csv',a,delimiter=',',comments='',
               header='time_fs,rg_A,cn,target_rg_A,target_cn,bias_eV,tip3p_eV,temperature_K')
    frames=read(out/'trajectory.traj',':')
    write(out/'trajectory.extxyz',frames)
    structural=[]
    for frame in frames:
        structural.append([frame.info['time_fs']]+[oxygen_rmsd(frame,refs[n]) for n in names])
    structural=np.array(structural)
    np.savetxt(out/'oxygen-rmsd.csv',structural,delimiter=',',comments='',header='time_fs,'+','.join(names))
    report=dict(metatomic_commit='b02b9ff3df6458cca21fa8690d229bbda97e6f78',kernel=str(KERNEL),
                reference_CVs=endpoints,temperature_K=150,timestep_fs=.5,stage_steps=stages,
                kappa=[30,5],max_bond_error_A=bond_error,
                structural_metric='Oxygen RMSD, minimum over permutations and proper rotations; does not classify H-bond topology',
                stages=[])
    for step in stages:
        idx=np.argmin(abs(structural[:,0]-step*.5))
        rmsd=dict(zip(names,structural[idx,1:].tolist()))
        report['stages'].append(dict(step=step,time_fs=step*.5,cv=a[step,1:3].tolist(),
                                    oxygen_rmsd_A=rmsd,nearest_reference=min(rmsd,key=rmsd.get)))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['stages'],indent=2),flush=True)
    print('PASS: CVs, target schedule, bias energies and rigid water constraints.')


if __name__=='__main__': main()

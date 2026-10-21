"""Load the source-built PLUMED kernel and pinned core plugin explicitly."""
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
KERNEL = HERE.parents[2] / 'plumed2/src/lib/libplumedKernel.dylib'
os.environ['PLUMED_KERNEL'] = str(KERNEL)
_registered = False


def cv_input(length_unit="A"):
    global _registered
    library = '' if _registered else f' PLUGIN_LIBRARY={HERE / "hexamer.dylib"}'
    species = ' SPECIES1=1,4,7,10,13,16 SPECIES2=2,3,5,6,8,9,11,12,14,15,17,18 SPECIES_TO_TYPES=8,1'
    result = [
        f'UNITS LENGTH={length_unit} ENERGY=eV TIME=fs',
        'rg: METATOMIC MODEL=oxygen-rg PLUGIN=water-hexamer' + library + species,
        'cn: METATOMIC MODEL=oxygen-cn PLUGIN=water-hexamer' + species,
    ]
    _registered = True
    return result


class BiasProbe:
    def __init__(self, cv, logfile, length_unit="A", extra_input=()):
        from plumed import Plumed
        from ase import units
        self.plumed = Plumed(kernel=str(KERNEL))
        for name, value in [('setMDEnergyUnits', units.mol / units.kJ),
                            ('setMDLengthUnits', 0.1), ('setMDTimeUnits', 0.001),
                            ('setMDMassUnits', 1.0), ('setMDChargeUnits', 1.0),
                            ('setNatoms', 18), ('setMDEngine', 'gradient-check'),
                            ('setLogFile', str(logfile)), ('setTimestep', 1.0)]:
            self.plumed.cmd(name, value)
        self.plumed.cmd('init')
        for line in cv_input(length_unit) + list(extra_input) + [f'bias: RESTRAINT ARG={cv} AT=0 KAPPA=1']:
            self.plumed.cmd('readInputLine', line)
        self.step = 0

    def evaluate(self, positions):
        p = np.array(positions, dtype=float, order='C')
        forces, virial = np.zeros_like(p), np.zeros((3, 3))
        masses = np.tile([15.999, 1.008, 1.008], 6)
        box = np.zeros((3, 3))
        self.plumed.cmd('setStep', self.step)
        self.plumed.cmd('setPositions', p)
        self.plumed.cmd('setBox', box)
        self.plumed.cmd('setMasses', masses)
        self.plumed.cmd('setEnergy', 0.0)
        self.plumed.cmd('setForces', forces)
        self.plumed.cmd('setVirial', virial)
        self.plumed.cmd('calc')
        energy = np.zeros(1)
        self.plumed.cmd('getBias', energy)
        self.step += 1
        return energy[0], forces, virial

    def close(self):
        self.plumed.finalize()

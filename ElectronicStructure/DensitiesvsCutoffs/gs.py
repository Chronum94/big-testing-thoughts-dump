import numpy as np

from gpaw import GPAW, PW, MarzariVanderbilt, Mixer
from ase.build import bulk
from ase.parallel import paropen


atoms = bulk("NaCl")
atoms.rattle()
densities = []
forces = []
energies = []
stresses = []
for ecut in np.arange(350, 951, 100):
    calc = GPAW(mode=PW(ecut),
        kpts=[6] * 3,
        occupations=MarzariVanderbilt(0.05),
        mixer=Mixer(0.4),
        h=0.12,
        convergence={"density": 1e-7})
    atoms.calc = calc
    energies.append(atoms.get_potential_energy())
    forces.append(atoms.get_forces())
    stresses.append(atoms.get_stress())
    densities.append(calc.get_pseudo_density())
with paropen(f"c_arrays.npz", "wb") as fd:
    np.savez(fd, energies=np.array(energies),
                forces=np.array(forces),
                stresses=np.array(stresses),
                densities=np.array(densities))

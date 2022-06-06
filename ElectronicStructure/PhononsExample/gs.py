from ase.build import bulk
from ase.io import write, read
from ase.parallel import parprint, paropen, world
from ase.optimize import BFGS
from gpaw import GPAW, PW, Mixer
import numpy as np


eq_energy = 0
energies = []
forces = []
hlist = np.linspace(-0.1, 0.1, 21)
for h in hlist:
    atoms = read("kcaf3_opt.xyz")
    # atoms = bulk("Si")
    calc = GPAW(mode=PW(350),
                kpts={'density': 3},
                occupations={'name': 'tetrahedron-method'},
                mixer=Mixer(0.5),
                # convergence={},
                txt=None,
                )

    atoms.positions[0] += np.array([h, 0, 0])

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    force = atoms.get_forces()
    energies.append(energy)
    forces.append(force)
    parprint(energy)

hlist = hlist[:]
energies = np.array(energies)
forces = np.array(forces)

if world.rank == 0:
    np.savez("datakcaf3_2.npz", hlist=hlist, energies=energies, forces=forces)

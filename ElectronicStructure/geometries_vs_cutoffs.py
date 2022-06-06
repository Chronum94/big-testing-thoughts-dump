from gpaw import GPAW, PW, Mixer

from ase.build import molecule
from ase.optimize import BFGS
from ase.io import write

import numpy as np


for ecut in range(400, 1601, 200):

    atoms = molecule("H2O")
    atoms.center(vacuum=5)

    calc = GPAW(
        mode=PW(ecut),
        convergence={"forces": 1e-7},
        mixer=Mixer(0.5),
    )
    atoms.calc = calc

    opt = BFGS(atoms)
    opt.run(fmax=1e-5)
    write(f"atoms_{ecut:04d}.xyz", atoms)

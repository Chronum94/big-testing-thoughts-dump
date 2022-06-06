from ase.build import bulk
from ase.io import write, read
from ase.parallel import parprint, paropen, world
from ase.optimize import BFGS
from gpaw import GPAW, PW, Mixer
import numpy as np


atoms = read("kcaf3.xyz")
calc = GPAW(mode=PW(350),
            kpts={'density': 3},
            occupations={'name': 'tetrahedron-method'},
            mixer=Mixer(0.5),
            # convergence={},
            txt="kcaf3_opt.log",
            symmetry={"point_group": False}
            )

atoms.calc = calc
atoms.rattle(seed=263457547)
opt = BFGS(atoms, alpha=2)
opt.run(fmax=1e-3)

write("kcaf3_opt.xyz", atoms)

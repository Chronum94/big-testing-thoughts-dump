from gpaw import GPAW, FermiDirac, PW, Mixer, setup_paths
from ase.io import read
from ase.optimize import BFGS
from ase.phonons import Phonons
from ase.parallel import paropen
import numpy as np

setup_paths.insert(0, '.')

atoms = read("libr03.txt")

calc = GPAW(mode=PW(650),  # energy cutoff for plane wave basis (in eV)
            kpts={'size': (6, 6, 6)},
            xc='PBE',
            mixer=Mixer(0.4),
            convergence={'forces': 1e-5},
            occupations=FermiDirac(0.0),
            random=True,
            txt='libr03ph.txt',
            setups={'Li': 'sv'},
            symmetry={'point_group': False}
           )

ph = Phonons(atoms, calc=calc, supercell=(1, 1, 1), delta=0.01, name="libr03ph")
ph.run()
ph.read(acoustic=True, symmetrize=5)
energies = ph.band_structure([[0, 0, 0]], born=False, verbose=False)
with paropen(f"phononfreqs_libr03.csv", "w") as fd:
    np.savetxt(fd, energies * 241.8, delimiter=",")

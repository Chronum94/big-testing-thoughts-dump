from gpaw import GPAW, FermiDirac, PW, Mixer, setup_paths
from ase.io import read
from ase.optimize import BFGS
from ase import Atoms

setup_paths.insert(0, '.')

atoms = read("LiBr_mp-976280_computed.cif")

calc = GPAW(mode=PW(650),  # energy cutoff for plane wave basis (in eV)
            kpts={'size': (6, 6, 6)},
            xc='PBE',
            mixer=Mixer(0.4),
            convergence={'forces': 1e-5},
            occupations=FermiDirac(0.0),
            random=True,
            txt='libr03.txt',
            setups={'Li': 'sv'}
           )

atoms.calc = calc
opt = BFGS(atoms, logfile='optlibr.out')
opt.run(fmax=1e-4)

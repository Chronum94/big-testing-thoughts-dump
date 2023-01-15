from gpaw import GPAW
from ase.build import bulk
import numpy as np
from ase.parallel import paropen


atoms = bulk("Mo")

calc = GPAW(gpts=(30, 30, 30), mode='pw', kpts=[4, 4, 4])
atoms.calc = calc
atoms.get_potential_energy()

pspw = calc.get_all_electron_density(gridrefinement=4)

calc = GPAW(gpts=(30, 30, 30), mode='lcao', kpts=[4, 4, 4], basis='sz(dzp)')
atoms.calc = calc
atoms.get_potential_energy()

pslcao = calc.get_all_electron_density(gridrefinement=4)

with paropen("mo_pspw.npz", "wb") as fd:
    np.savez(fd, rho=pspw)

with paropen("mo_pslcao.npz", "wb") as fd:
    np.savez(fd, rho=pslcao)

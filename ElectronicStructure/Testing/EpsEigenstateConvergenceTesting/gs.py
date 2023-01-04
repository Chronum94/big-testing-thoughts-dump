from gpaw import GPAW, PW, FermiDirac, Mixer
from ase.io import read, write
from ase.parallel import parprint

import numpy as np


atoms = read("TaS2_scell_geom_spair_05eTa-650eV-fullrelaxed_shifted_fineconvg.xyz")


calc = GPAW(
    atoms=atoms,
    mixer=Mixer(beta=0.4),
    xc="PBE",
    mode=PW(350),
    kpts={"size": [4, 4, 12],},
    occupations=FermiDirac(0.1),
    txt="outgs.txt",
    setups={"Ta": "5:d,2.27,0"},
)

atoms.get_potential_energy()
calc.write("outgs.gpw")

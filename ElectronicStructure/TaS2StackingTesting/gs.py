from gpaw import GPAW, PW, MarzariVanderbilt, Mixer
from ase.io import read, write
from ase.build import stack
import sys
import numpy as np


jobid = int(sys.argv[1])
jobid -= 1
atoms = read(f"layers.extxyz@{jobid}")

calc = GPAW(mode=PW(350),
           mixer=Mixer(0.3),
           occupations=MarzariVanderbilt(0.1),
            kpts = [4, 4, 6],
            nbands = "140%",
            # convergence={'density': 1e-3, 'eigenstates': 1e-5},
            parallel={'sl_auto': True},
            setups={"Ta": "5:d,2.27,0"},
            txt=f"atoms{jobid}.txt",
           )

atoms.calc = calc
atoms.get_potential_energy()

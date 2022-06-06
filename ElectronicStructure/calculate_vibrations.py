from gpaw import GPAW, PW, Mixer
from ase.io import write, read

from ase.vibrations import Vibrations

import numpy as np


for ecut in range(400, 1601, 200):
    for negative_force_convergence_log in range(2, 6):

        atoms = read(f"atoms_{ecut:04d}.xyz")

        filename = f"ecut_{ecut:04d}/logforcconv_{negative_force_convergence_log}"

        calc = GPAW(
            mode=PW(ecut),
            convergence={
                "forces": 10 ** (-negative_force_convergence_log),
                "eigenstates": 10 ** (-negative_force_convergence_log),
                "density": 10 ** (-negative_force_convergence_log),
            },
            mixer=Mixer(0.5),
            symmetry={"point_group": False},
        )
        atoms.calc = calc

        vib = Vibrations(atoms, name=filename)
        vib.run()
        energies = vib.get_frequencies(method="Frederiksen")
        np.savetxt(filename + "freqs.csv", energies)

"""Induced-charge profile between two constant-charge SJM runs."""
import numpy as np
from ase.units import Bohr
import bench_setup as bs

profiles = {}
for n in [0.0, 0.03]:
    atoms, calc = bs.make(f'ind{n}', {'excess_electrons': n,
                                       'target_potential': None})
    atoms.get_potential_energy()
    gd = calc.density.finegd
    nt = gd.collect(calc.density.nt_sg.sum(0), broadcast=True).mean(0).mean(0)
    eps = gd.collect(calc.hamiltonian.dielectric.eps_gradeps[0],
                     broadcast=True).mean(0).mean(0)
    jm = gd.collect(calc.density.background_charge.mask_g,
                    broadcast=True).mean(0).mean(0)
    profiles[n] = nt, calc.get_electrode_potential()
z = gd.coords(2, pad=False) * Bohr
dn = profiles[0.03][0] - profiles[0.0][0]
ztop = atoms.positions[:, 2].max()
sel = z > ztop - 1.2  # outer half of the top layer and beyond, minus jellium
sel &= jm == 0
print('top atom', ztop, 'layer spacing',
      np.diff(sorted(set(np.round(atoms.positions[:, 2], 3)))))
print('induced centroid (z>ztop-1.2, no jellium):',
      (z[sel] * dn[sel]).sum() / dn[sel].sum())
print('first eps>1.01 at', z[(eps > 1.01) & (z > ztop)][0])
print('eps 50% at', z[(eps > 0.5 * (1 + eps.max())) & (z > ztop)][0])
print('jellium centroid', (z * jm).sum() / jm.sum())
slope = (profiles[0.03][1] - profiles[0.0][1]) / 0.03
print('slope', slope)
np.savetxt('induced.txt', np.c_[z, dn, eps, jm, profiles[0.0][0]])

import numpy as np
from ase.units import Bohr, Ha
import bench_setup as bs
out = {}
for n in [0.0, 0.03]:
    atoms, calc = bs.make(f'ind2_{n}', {'excess_electrons': n,
                                        'target_potential': None})
    atoms.get_potential_energy()
    gd = calc.density.finegd
    v = gd.collect(calc.hamiltonian.vHt_g, broadcast=True).mean(0).mean(0) * Ha
    rhot = gd.collect(calc.density.rhot_g, broadcast=True).mean(0).mean(0)
    out[n] = (v, calc.get_fermi_level(), calc.get_electrode_potential(), rhot)
z = gd.coords(2, pad=False) * Bohr
dv = out[0.03][0] - out[0.0][0]
print('d(E_F)', out[0.03][1] - out[0.0][1], 'd(phi)', out[0.03][2] - out[0.0][2])
print('dv at bulk (z=8.35)', dv[np.argmin(abs(z - 8.35))], 'dv top', dv[-1], 'dv bottom', dv[0])
np.savetxt('induced2.txt', np.c_[z, dv, out[0.03][3] - out[0.0][3]])

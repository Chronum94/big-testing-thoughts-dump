import numpy as np
from ase.units import Bohr, _e, Ha
from ase.build import fcc111
z, dn, eps, jm, n0 = np.loadtxt('induced.txt').T
h = (z[1] - z[0]) / Bohr
a = fcc111('Au', size=(1, 1, 3))
A = abs(np.linalg.det(a.cell[:2, :2])) / Bohr**2
ztop = 10.711
conv = 1 / Ha / (A * Bohr**2) * _e * 1e22


def cap(delec):
    """delec: electron excess profile (per bohr, xy-integrated), sum=1."""
    rho = -delec / (delec.sum() * h) + jm / (jm.sum() * h)  # charge, +1 jellium
    q = np.cumsum(rho) * h  # enclosed charge (e) from the bottom
    dphi = 4 * np.pi / A * (q / eps).sum() * h  # Ha per electron
    return 1 / dphi * conv


dn_line = dn * A  # nt is per bohr^3; xy-average * A
print('actual induced dn:', cap(np.where(jm == 0, dn_line, 0)))
grad = -np.gradient(n0, h)
for zcut in [ztop - 1.2, ztop - 0.5, ztop, ztop + 0.3]:
    m = np.where((z > zcut) & (grad > 0) & (jm == 0), grad, 0)
    print('rigid-shift model, z >', round(zcut, 2), cap(m))
print('true 36.0')

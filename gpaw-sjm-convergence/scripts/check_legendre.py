import json
import numpy as np
from scipy.interpolate import CubicSpline

for el, target in [('Au', 4.0), ('Pt', 4.5)]:
    n, om, phi = np.array(json.load(open(f'sweep_{el}.json'))).T
    order = np.argsort(phi)
    # Thermodynamic consistency: dOmega/dphi should equal N (mu = -phi).
    dom = np.gradient(om, phi)
    print(f'{el}: dOmega/dphi vs N (rms diff):',
          np.sqrt(np.mean((dom[1:-1] - n[1:-1])**2)).round(4),
          ' N range', n.min(), n.max())
    ref = CubicSpline(phi[order], om[order])(target)
    c = -np.gradient(n, phi)  # e/V, local capacitance
    print(f'  reference Omega(phi*={target}) = {ref:.5f} eV')
    print('   phi-phi*   raw err   1st-order err   2nd-order err  (meV)')
    for ni, oi, pi, ci in zip(n, om, phi, c):
        dmu = pi - target
        o1 = oi - ni * dmu
        o2 = o1 - 0.5 * ci * dmu**2
        print(f'   {pi - target:+.3f}   {1e3 * (oi - ref):+9.2f}   '
              f'{1e3 * (o1 - ref):+9.3f}   {1e3 * (o2 - ref):+9.3f}')

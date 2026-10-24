"""Constant-charge sweep: Omega(N) and phi(N), to check the Legendre
extrapolation Omega(mu*) = Omega(mu) - N dmu - C dmu^2 / 2."""
import json
import sys

import numpy as np
import bench_setup

element = sys.argv[1]
ns = np.round(np.arange(float(sys.argv[2]), float(sys.argv[3]) + 1e-9, 0.01), 4)
atoms, calc = bench_setup.make(f'sweep_{element}', {'target_potential': None,
                                                    'excess_electrons': ns[0]},
                               element=element)
rows = []
for n in ns:
    calc.set(sj={'excess_electrons': float(n)})
    omega = atoms.get_potential_energy()
    rows.append((float(n), omega, calc.get_electrode_potential()))
    print(rows[-1], flush=True)
json.dump(rows, open(f'sweep_{element}.json', 'w'))

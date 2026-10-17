"""Smoke test: does Gibbs sampling of the *exact* bond-RBM (always accepted) mix faster than
single-flip Metropolis near T_c? tau in units of one full-lattice update each."""
import sys
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, metropolis_sweep
from rbm import exact_ising_rbm
from run import tau_int

T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.269
beta, n = 1 / T, 20000
for L in (4, 8, 16):
    rng = np.random.default_rng(L)
    lat = Lattice(L, 2)
    rbm = exact_ising_rbm(lat, beta)
    out = {}
    for name in ("single-flip", "exact-RBM gibbs"):
        s = rng.integers(2, size=lat.N)
        m = []
        for t in range(n + 2000):
            if name == "single-flip":
                metropolis_sweep(lat, s, beta, rng)
            else:
                s = rbm.gibbs(s, rng)
            if t >= 2000:
                m.append(lat.order_parameter(s)[0])
        m = np.array(m)
        out[name] = (tau_int(m), m.mean())
    print(f"L={L:3d} " + "  ".join(f"{k}: tau_|m|={v[0]:6.1f} <|m|>={v[1]:.3f}" for k, v in out.items()), flush=True)

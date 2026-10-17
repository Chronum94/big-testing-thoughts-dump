"""Idea #1: dynamic scaling at T_c (2D Ising) of Metropolis, exact bond-RBM Gibbs, Wolff.
Cost-fair metric: tau_|m| (in that sampler's own updates) x ms per update."""
import sys
import time
import warnings
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, metropolis_sweep, wolff_step
from rbm import exact_ising_rbm
from run import tau_int

beta, n = 1 / 2.269185, 20000
samplers = {
    "metropolis": lambda lat, s, rng, rbm: (metropolis_sweep(lat, s, beta, rng), s)[1],
    "exactRBM":   lambda lat, s, rng, rbm: rbm.gibbs(s, rng),
    "wolff":      lambda lat, s, rng, rbm: (wolff_step(lat, s, beta, rng), s)[1],
}
for L in (8, 16, 24):
    lat, rng = Lattice(L, 2), np.random.default_rng(L)
    rbm = exact_ising_rbm(lat, beta)
    row = [f"L={L:3d}"]
    for name, step in samplers.items():
        s = rng.integers(2, size=lat.N)
        for _ in range(2000):
            s = step(lat, s, rng, rbm)
        m, t0 = np.empty(n), time.perf_counter()
        for t in range(n):
            s = step(lat, s, rng, rbm)
            m[t] = lat.order_parameter(s)[0]
        ms = 1e3 * (time.perf_counter() - t0) / n
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tau = tau_int(m)
        row.append(f"{name}: tau={tau:7.1f}{'!' if w else ' '} ms={ms:.3f} tau*ms={tau*ms:7.2f}")
    print("  ".join(row), flush=True)
print("! = run too short for reliable tau (n < 50 tau)")

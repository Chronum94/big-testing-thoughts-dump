"""Is the RBM speed-up at small L just global sector switching? Potts q=3, 2D L=6, T=1.0 (~T_c).
Arms (each step = 1 Metropolis sweep + one extra move): single-flip, +RBM, +random global colour
permutation (exact symmetry, always accepted). 8 independent chains per arm; <E>/N +- sem, tau."""
import sys
import warnings
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, generate_dataset, metropolis_sweep
from rbm import RBM, local_mask, rbm_mh_step
from run import tau_int

warnings.simplefilter("ignore")
# usage: potts_control.py q T L radius block   (radius/block -1 = dense / full-system)
q, T, L, radius, block = (int(sys.argv[1]), float(sys.argv[2]), *map(int, sys.argv[3:6])) if len(sys.argv) > 3 else (3, 1.0, 6, -1, -1)
n = 8000
beta, rng = 1 / T, np.random.default_rng(0)
lat = Lattice(L, 2, q=q)
data = generate_dataset(lat, beta, 1500, thin=5, rng=rng)
rbm = RBM(lat.N, q, lat.N, mask=local_mask(lat, 1, radius) if radius >= 0 else None, rng=rng).fit(data, epochs=20, lr=0.05, k=5, rng=rng)

def relabel(s, rng):
    return rng.permutation(q)[s]

def block_sites(rng):
    return np.arange(lat.N) if block < 0 else np.flatnonzero(lat.distance(np.arange(lat.N), rng.integers(lat.N)) <= block)


def metro_on_block(s, rng):
    """Work-matched null: one Metropolis attempt per site of the same kind of block (a full sweep if block < 0)."""
    for i in rng.permutation(block_sites(rng)):
        new = (s[i] + rng.integers(1, q)) % q
        dE = lat.delta_energy(s, i, new)
        if dE <= 0 or rng.random() < np.exp(-beta * dE):
            s[i] = new
    return s


arms = {"single-flip": None,
        "+metro-block": metro_on_block,
        "+RBM": lambda s, rng: rbm_mh_step(rbm, lat, beta, s, rng, None if block < 0 else
                                           np.flatnonzero(lat.distance(np.arange(lat.N), rng.integers(lat.N)) <= block))[0],
        "+relabel": relabel}
for name, extra in arms.items():
    Em, tE, tm = [], [], []
    for c in range(8):
        r = np.random.default_rng(10 + c)
        s = data[r.integers(len(data))].copy()
        E, m = np.empty(n), np.empty(n)
        for t in range(n):
            metropolis_sweep(lat, s, beta, r)
            if extra is not None:
                s = extra(s, r)
            E[t], m[t] = lat.energy(s) / lat.N, lat.order_parameter(s)[0]
        Em.append(E.mean()); tE.append(tau_int(E)); tm.append(tau_int(m))
    print(f"{name:12s} <E>/N = {np.mean(Em):.4f} +- {np.std(Em, ddof=1)/np.sqrt(8):.4f}   "
          f"tau_E = {np.mean(tE):5.1f} +- {np.std(tE, ddof=1)/np.sqrt(8):.1f}   tau_m = {np.mean(tm):5.1f} +- {np.std(tm, ddof=1)/np.sqrt(8):.1f}", flush=True)

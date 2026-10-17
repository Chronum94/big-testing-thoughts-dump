"""Round-2 #1: lambda-bridge, 2D Ising at T_c. Gates K = 2 lam beta |J| + exact residual RBM at
(1-lam) beta -> acceptance 1 for all lam; lam=1 is Swendsen-Wang. tau_|m| per gated step, 3 seeds.
lam=0 reference (pure exact-RBM Gibbs): NOTES section 7."""
import sys
import warnings
import numpy as np
sys.path.insert(0, '.')
from model import Lattice
from rbm import exact_ising_rbm, gate_strengths, gated_mh_step
from run import tau_int

warnings.simplefilter("ignore")
beta, n = 1 / 2.269185, 6000
for L in (8, 16):
    lat = Lattice(L, 2)
    for lam in (0.5, 0.8, 0.9, 0.95, 1.0):
        rbm, K = exact_ising_rbm(lat, (1 - lam) * beta), gate_strengths(lat, beta, lam)
        taus, acc, sizes = [], 0, []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            s = rng.integers(2, size=lat.N)
            m = np.empty(n)
            for t in range(n + 500):
                s, a, c = gated_mh_step(rbm, lat, beta, K, s, rng)
                if t >= 500:
                    m[t - 500] = abs((2 * s - 1).mean()); acc += a; sizes.append(c)
            taus.append(tau_int(m))
        print(f"L={L:2d} lam={lam:4.2f}  tau_|m| = {np.mean(taus):7.1f} +- {np.std(taus)/np.sqrt(3):5.1f}"
              f"  acc={acc/(3*n):.3f}  <changed>/N={np.mean(sizes)/lat.N:.2f}", flush=True)

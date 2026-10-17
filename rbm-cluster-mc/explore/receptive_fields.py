"""Idea #5-lite: what do hidden units of a dense RBM learn on 2D Ising? Spin-flip weight
u_m = W[:,1,m] - W[:,0,m]. Measure: top singular values of U, |cos| of top singular vector with
the uniform (magnetisation) mode, and localisation of each u_m (IPR*N: 1=uniform, N=one site).
With augment=True (Z2 symmetric data) a uniform mode must appear as +/- pairs or via |cos|."""
import sys
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, generate_dataset
from rbm import RBM

L = 8
for T in (1.8, 2.269, 3.5):
    rng = np.random.default_rng(0)
    lat = Lattice(L, 2)
    data = generate_dataset(lat, 1 / T, 1500, thin=5, rng=rng)
    rbm = RBM(lat.N, 2, 32, rng=rng).fit(data, epochs=40, lr=0.05, k=5, rng=rng)
    U = rbm.W[:, 1, :] - rbm.W[:, 0, :]               # (N, M)
    u, sv, _ = np.linalg.svd(U, full_matrices=False)
    cos_uniform = abs(u[:, 0].sum()) / np.sqrt(lat.N)
    ipr = (U**4).sum(0) / (U**2).sum(0)**2 * lat.N
    norms = np.linalg.norm(U, axis=0)
    print(f"T={T:5.3f}  sv[:4]={np.round(sv[:4], 2)}  |cos(top mode, uniform)|={cos_uniform:.3f}  "
          f"IPR*N median={np.median(ipr):5.1f}  |u_m| max/median={norms.max()/np.median(norms):.1f}", flush=True)

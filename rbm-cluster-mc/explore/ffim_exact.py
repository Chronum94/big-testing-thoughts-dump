"""Exact <E>/N for the periodic L x L Villain FFIM (model.make_bonds kind="ffim") by column transfer matrix.
Columns = axis-0 coordinate x; in-column (axis-1) bonds have sign (-1)^x; inter-column bonds ferro."""
import sys
import itertools
import numpy as np


def logZ(L, beta):
    sig = 2 * np.array(list(itertools.product([0, 1], repeat=L))) - 1        # (2^L, L) column states
    inter = sig @ sig.T                                                       # sum_y s(x,y) s(x+1,y)
    intra = (sig * np.roll(sig, -1, axis=1)).sum(1)                           # sum_y s(x,y) s(x,y+1)
    T = np.eye(len(sig))
    log_scale = 0.0
    for x in range(L):
        T = T @ (np.exp(beta * (-1) ** x * intra)[:, None] * np.exp(beta * inter))
        m = np.abs(T).max(); T /= m; log_scale += np.log(m)
    return np.log(np.trace(T)) + log_scale


L = int(sys.argv[1]) if len(sys.argv) > 1 else 8
for T in (1.0, 0.5):
    b, h = 1 / T, 1e-5
    E = -(logZ(L, b + h) - logZ(L, b - h)) / (2 * h)
    print(f"L={L} T={T}: exact <E>/N = {E / L**2:.6f}")

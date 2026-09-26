"""SV spectra / effective rank of core C, pair tensor W (present pairs), embeddings u, v."""
import sys
from itertools import combinations_with_replacement as cwr
import numpy as np
from ase.io import read
from apax.train.checkpoints import restore_parameters

def eff_rank(s):  # Roy & Vetterli entropy effective rank
    p = s / s.sum(); p = p[p > 0]
    return np.exp(-(p * np.log(p)).sum())

def report(name, M):
    s = np.linalg.svd(M, compute_uv=False)
    n99 = int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), 0.99) + 1)
    print(f"  {name:22s} shape {str(M.shape):12s} eff_rank {eff_rank(s):5.2f}  n(99% energy) {n99:3d}  "
          f"s/s0: {' '.join(f'{x:.2f}' for x in (s / s[0])[:10])}")

A = read(sys.argv[1], ":")
for d in sys.argv[2:]:
    _, params = restore_parameters(f"models/{d}")
    flat = {"/".join(k): np.asarray(v) for k, v in __import__("flax").traverse_util.flatten_dict(params).items()}
    get = lambda n: next(v for k, v in flat.items() if k.endswith(n))
    C = get("pair_core"); u = get("pair_emb_centre"); v = get("pair_emb_nbr")
    n_basis, KR = C.shape; K = u.shape[1]; R = KR // K
    idx = np.load(f"models/{d}/train_val_idxs.npz")["train_idxs"]
    Z = sorted({z for i in idx for z in A[i].numbers})
    pairs = sorted({p for i in idx for p in cwr(sorted(set(A[i].numbers)), 2)})
    Ck = C.reshape(n_basis, K, R).transpose(1, 0, 2).reshape(K, -1)             # (K, B*R)
    # W for ordered (centre, nbr) pairs present in training
    W = np.stack([np.einsum("k,k,kx->x", u[a], v[b], Ck) for a, b in pairs] +
                 [np.einsum("k,k,kx->x", u[b], v[a], Ck) for a, b in pairs if a != b])
    print(f"{d}: rank {K}, {len(Z)} elements, {len(pairs)} unordered pairs in train")
    report("core C (K x B*R)", Ck)
    report("W pairs (n x B*R)", W)
    report("W pairs, mean removed", W - W.mean(0))
    report("u[present] (n_el x K)", u[Z]); report("v[present] (n_el x K)", v[Z])
    report("u - 1", u[Z] - 1.0); report("v - 1", v[Z] - 1.0)

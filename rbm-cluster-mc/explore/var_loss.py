"""Idea #4: train the RBM by minimising Var_data[d], d = -beta E + F_RBM, vs PCD.

dVar/dtheta = 2 Cov_data(d, dF/dtheta); no negative phase. Risk: Var only constrains
p_RBM/pi on the data support, so the RBM may put mass off-manifold -> check acceptance.
"""
import copy
import sys
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, generate_dataset
from rbm import RBM, sigmoid, log_ratio, rbm_mh_step
from run import run_chain, tau_int


def var_fit(rbm, lat, beta, data, epochs, lr, batch=64, rng=None):
    """Plain SGD on Var_batch[d]. Per-sample dF: dF/dW = -onehot*ph, dF/db = -onehot, dF/dc = -ph."""
    rng = np.random.default_rng(rng)
    E = lat.energy(data)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(data)), max(1, len(data) // batch)):
            x = data[idx]
            onehot, ph = np.eye(rbm.q)[x], sigmoid(rbm.hidden_field(x))
            d = -beta * E[idx] + rbm.free_energy(x)
            w = 2 * (d - d.mean()) / len(x)                  # dVar/dF_b
            rbm.W -= lr * -np.einsum("b,bia,bm->iam", w, onehot, ph) * rbm.mask
            rbm.b -= lr * -np.einsum("b,bia->ia", w, onehot)
            rbm.c -= lr * -(w @ ph)
    return rbm


def moved(rbm, lat, beta, s, rng, block, n=1000):
    mv, sz = 0, 0
    for _ in range(n):
        sites = None if block < 0 else np.flatnonzero(lat.distance(np.arange(lat.N), rng.integers(lat.N)) <= block)
        s, _, c = rbm_mh_step(rbm, lat, beta, s, rng, sites)
        mv += c > 0
        sz += c
    return mv / n, sz / max(mv, 1)


def evaluate(rbm, lat, beta, test, rng):
    props = rbm.gibbs(test, rng)
    r = dict(sd_data=log_ratio(rbm, lat, beta, test).std(),
             sd_prop=log_ratio(rbm, lat, beta, props).std(),
             # mean shift of d on proposals vs data: negative => proposals sit in low-pi/p_RBM regions
             dshift=log_ratio(rbm, lat, beta, props).mean() - log_ratio(rbm, lat, beta, test).mean())
    for blk in (-1, 1, 2):
        r[f"mv{blk}"], r[f"C{blk}"] = moved(rbm, lat, beta, test[0].copy(), rng, blk)
    return r


def main(L, d, kind, T, seeds, tau_steps, lr_var=0.01, lr_ft=0.001):
    beta = 1 / T
    rows = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        lat = Lattice(L, d, kind=kind, rng=rng)
        data = generate_dataset(lat, beta, 2000, thin=5, rng=rng)
        tr, te = data[:1500], data[1500:]
        init = RBM(lat.N, 2, lat.N, rng=seed)
        pcd = copy.deepcopy(init).fit(tr, epochs=30, lr=0.05, k=5, rng=seed)
        arms = {"PCD": pcd,
                "Var": var_fit(copy.deepcopy(init), lat, beta, tr, 30, lr_var, rng=seed),
                "PCD+Var": var_fit(copy.deepcopy(pcd), lat, beta, tr, 30, lr_ft, rng=seed)}
        for name, rbm in arms.items():
            r = evaluate(rbm, lat, beta, te, rng)
            if tau_steps:
                c = run_chain(lat, beta, te[-1], tau_steps, rng, rbm=rbm, n_rbm=4, block=2)
                r["tauE"] = tau_int(c["E"])
            rows.setdefault(name, []).append(r)
            print(seed, name, {k: round(float(v), 3) for k, v in r.items()}, flush=True)
        if tau_steps:
            c = run_chain(lat, beta, te[-1], tau_steps, rng)
            rows.setdefault("single-flip", []).append({"tauE": tau_int(c["E"])})
    print(f"\n{L}x{L} {kind} T={T}: mean ± std over seeds {list(seeds)}")
    for name, rs in rows.items():
        print(f"{name:12s} " + "  ".join(f"{k}={np.mean([r[k] for r in rs]):.3f}±{np.std([r[k] for r in rs]):.3f}" for k in rs[0]))


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    which = sys.argv[1] if len(sys.argv) > 1 else "ising"
    # lrs picked by a 1-seed scan (Var-only: 0.03 diverges; fine-tune 3e-4..3e-3 all similar)
    if which == "ising":
        main(8, 2, "uniform", 2.5, range(3), tau_steps=3000)
    else:
        main(6, 2, "pm", 1.0, range(3), tau_steps=3000)

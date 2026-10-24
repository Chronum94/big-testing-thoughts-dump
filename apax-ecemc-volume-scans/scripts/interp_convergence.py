"""How many grid points does a tabulated radial need? Cubic Hermite (exact node values + derivatives) vs plain cubic
spline (values only), errors on a fine grid, vs number of nodes.

usage: python interp_convergence.py MODEL_DIR OUT.png [R_MIN]   (CPU, float64)
Covers every (centre, neighbour) pair of the elements below; outputs = n_radial channels and the radial_compression
(tensor) channels. Error = max over pairs, channels and fine-grid points, relative to the max |R| (or |dR/dr|) of that
pair and channel, reported both over [R_MIN, r_max] and over [1.5 A, r_max] (where the training data lives).
"""
import sys

import jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/chronum/scratch/common"); import cpuload  # noqa: E401,F401,E402
import apax.train.checkpoints as ck  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E402,E702
import numpy as np  # noqa: E402
from apax.nn.builder import ModelBuilder  # noqa: E402
from scipy.interpolate import CubicHermiteSpline, CubicSpline  # noqa: E402

cfg, params = ck.restore_parameters(sys.argv[1]); r_min = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
mc = cfg.model.model_dump(); mc["descriptor_dtype"] = "fp64"  # float64 reference so the error floor is the interpolant's
radial = ModelBuilder(mc).build_radial_function()
rep = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), params["params"]["energy_model"]["representation"])
W = np.asarray(rep["radial_compression"]["kernel"]); r_max = radial.basis_fn.r_max
ELEM = [1, 3, 6, 8, 9, 15]  # H Li C O F P
pairs = [(c, n) for c in ELEM for n in ELEM]


def R_and_dR(d, c, n):
    f = lambda x: radial.apply({"params": rep["radial_fn"]}, x, jnp.full(x.shape, n), jnp.full(x.shape, c))  # noqa: E731
    R, dR = jax.jvp(f, (d,), (jnp.ones_like(d),))
    R, dR = np.asarray(R), np.asarray(dR)
    return np.concatenate([R, R @ W], 1), np.concatenate([dR, dR @ W], 1)


fine = np.linspace(r_min, r_max, 20001)
ref = [R_and_dR(jnp.asarray(fine), c, n) for c, n in pairs]
inner = fine >= 1.5
Ns = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192]
res = {k: [] for k in ("herm_R", "herm_dR", "spl_R", "spl_dR", "herm_R_in", "herm_dR_in")}
for N in Ns:
    nodes = np.linspace(r_min, r_max, N); e = {k: 0.0 for k in res}
    for (c, n), (Rf, dRf) in zip(pairs, ref):
        Rn, dRn = R_and_dR(jnp.asarray(nodes), c, n)
        sR, sdR = np.abs(Rf).max(0) + 1e-30, np.abs(dRf).max(0) + 1e-30
        H = CubicHermiteSpline(nodes, Rn, dRn, axis=0); S = CubicSpline(nodes, Rn, axis=0)
        for tag, P in (("herm", H), ("spl", S)):
            eR = np.abs(P(fine) - Rf) / sR; edR = np.abs(P(fine, 1) - dRf) / sdR
            e[f"{tag}_R"] = max(e[f"{tag}_R"], eR.max()); e[f"{tag}_dR"] = max(e[f"{tag}_dR"], edR.max())
            if tag == "herm":
                e["herm_R_in"] = max(e["herm_R_in"], eR[inner].max()); e["herm_dR_in"] = max(e["herm_dR_in"], edR[inner].max())
    for k in res:
        res[k].append(e[k])
    print(f"N {N:5d}  spacing {(r_max - r_min) / (N - 1):.4f} A  Hermite R {e['herm_R']:.2e} dR {e['herm_dR']:.2e} | "
          f">=1.5A R {e['herm_R_in']:.2e} dR {e['herm_dR_in']:.2e} | spline R {e['spl_R']:.2e} dR {e['spl_dR']:.2e}", flush=True)
h = (r_max - r_min) / (np.array(Ns) - 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for a, q, lab in ((ax[0], "R", "R"), (ax[1], "dR", "dR/dr (forces)")):
    a.loglog(Ns, res[f"herm_{q}"], "o-", color="#2a6fdb", label=f"cubic Hermite, [{r_min:g}, {r_max:g}] Å")
    a.loglog(Ns, res[f"herm_{q}_in"], "s--", color="#2a6fdb", alpha=0.6, label="cubic Hermite, [1.5, r_max] Å")
    a.loglog(Ns, res[f"spl_{q}"], "^-", color="#e0672b", label="cubic spline (values only)")
    p = 4 if q == "R" else 3; ref_line = res[f"herm_{q}"][4] * (h / h[4]) ** p
    a.loglog(Ns, ref_line, ":", color="grey", label=f"∝ h^{p}")
    for tol in (1e-4, 1e-6):
        a.axhline(tol, color="k", lw=0.5, ls="--")
    a.set(xlabel="interpolation nodes N", ylabel=f"max relative error in {lab}", title=f"{lab}: worst pair × channel")
    a.legend(fontsize=8)
fig.suptitle(f"{sys.argv[1].rstrip('/').split('/')[-1]}: 36 element pairs × {W.shape[0] + W.shape[1]} channels, float64 reference on 20001 points")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110); print("wrote", sys.argv[2])

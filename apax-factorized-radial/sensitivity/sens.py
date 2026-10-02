# Sensitivity of atom 0's energy/force to deleting or rattling its farthest neighbours.
import numpy as np, jax, jax.numpy as jnp
from ase.io import read
import apax
from apax.config.model_config import GMNNConfig
from apax.data.preprocessing import compute_nl
from apax.layers.scaling import PerElementScaleShift

import sys
L, f = (18.04, "carbon500.xyz") if "500" in sys.argv else (28.63, "carbon.xyz")
a = read(f); a.cell = [L] * 3; a.pbc = True; a.numbers[:] = 6
c = np.argmin(np.linalg.norm(a.positions - L / 2, axis=1))  # centre atom -> index 0
a = a[[c] + [i for i in range(len(a)) if i != c]]
d_all = a.get_distances(0, range(len(a)), mic=True)
print(f"N {len(a)}  min pair dist {a.get_all_distances(mic=True)[np.triu_indices(len(a), 1)].min():.3f} A")

cfg = GMNNConfig.model_validate({"name": "gmnn", "descriptor_dtype": "fp64", "readout_dtype": "fp64"}).model_dump()
rc = cfg["basis"]["r_max"]
box = np.asarray(a.cell.array)
builder = GMNNConfig.model_validate(cfg).get_builder()(cfg)
model = builder.build_energy_derivative_model(init_box=box)
emodel = builder.build_energy_model(init_box=box)


def inputs(at, gone=()):
    # fixed shapes: deleted atoms keep their slot with Z=0 and lose all pairs; NL padded to PMAX
    frac = at.get_scaled_positions()
    idx, off = compute_nl(frac, box, rc)
    keep = ~np.isin(idx, list(gone)).any(0)
    idx, off = idx[:, keep], off[keep]
    pad = PMAX - idx.shape[1]
    idx, off = np.pad(idx, ((0, 0), (0, pad))), np.pad(off, ((0, pad), (0, 0)))
    Z = at.numbers.copy(); Z[list(gone)] = 0
    return jnp.asarray(frac), jnp.asarray(Z), jnp.asarray(idx, jnp.int32), jnp.asarray(box), jnp.asarray(off)


PMAX = compute_nl(a.get_scaled_positions(), box, rc)[0].shape[1] + 2000


params = model.init(jax.random.PRNGKey(0), *inputs(a))


f_jit = jax.jit(lambda *x: model.apply(params, *x)["forces"][0])
e_jit = jax.jit(lambda *x: jax.tree_util.tree_leaves(emodel.apply({"params": params["params"]["energy_model"]}, *x, capture_intermediates=lambda m, _: isinstance(m, PerElementScaleShift), mutable=["intermediates"])[1])[0].reshape(-1)[0])


def ev(at, gone=()):
    x = inputs(at, gone)
    return float(e_jit(*x)), np.asarray(f_jit(*x))


E0, F0 = ev(a)
print(f"rc {rc}  n nbrs {int((d_all[1:] < rc).sum())}  E_0 {E0:.6f}  |F_0| {np.linalg.norm(F0):.6f}")
order = np.argsort(-np.where(d_all < rc, d_all, -1))[: int((d_all[1:] < rc).sum())]  # nbrs, farthest first

print("\n-- delete farthest n neighbours --")
print(f"{'n':>4} {'d range':>14} {'|dE_0|':>10} {'|dF_0|':>10} {'|dF|/|F|':>9}")
for n in [1, 2, 5, 10, 20, 40]:
    sel = order[:n]
    E, F = ev(a, sel)
    dF = np.linalg.norm(F - F0)
    print(f"{n:4d} {d_all[sel].min():6.3f}-{d_all[sel].max():6.3f} {abs(E - E0):10.2e} {dF:10.2e} {dF / np.linalg.norm(F0):9.2e}")

print("\n-- delete outside rc (E_0 must not move, F_0 can) --")
for lo, hi in [(rc, rc + 1)] + ([] if "500" in sys.argv else [(2 * rc - 1, 2 * rc), (2 * rc, 2 * rc + 1)]):
    sel = np.where((d_all >= lo) & (d_all < hi))[0]
    E, F = ev(a, sel)
    print(f"shell {lo:4.1f}-{hi:4.1f} ({len(sel):3d} atoms)  |dE_0| {abs(E - E0):.2e}  |dF_0| {np.linalg.norm(F - F0):.2e}")

print("\n-- rattle farthest n neighbours, RMS over 10 seeds --")
print(f"{'n':>4} {'sigma':>6} {'rms dE_0':>10} {'rms |dF_0|':>11} {'|dF|/|F|':>9}")
for n in [1, 5, 20]:
    for s in [0.01, 0.05, 0.1]:
        dE, dF = [], []
        for seed in range(10):
            b = a.copy()
            b.positions[order[:n]] += np.random.default_rng(seed).normal(0, s, (n, 3))
            E, F = ev(b)
            dE.append(E - E0); dF.append(np.linalg.norm(F - F0))
        rE, rF = np.sqrt(np.mean(np.square(dE))), np.sqrt(np.mean(np.square(dF)))
        print(f"{n:4d} {s:6.2f} {rE:10.2e} {rF:11.2e} {rF / np.linalg.norm(F0):9.2e}")
print("\nreference: rattle nearest neighbour, sigma 0.05")
dF = []
for seed in range(10):
    b = a.copy(); b.positions[order[-1]] += np.random.default_rng(seed).normal(0, 0.05, 3)
    dF.append(np.linalg.norm(ev(b)[1] - F0))
print(f"  nn d {d_all[order[-1]]:.3f}  rms |dF_0| {np.sqrt(np.mean(np.square(dF))):.2e}")

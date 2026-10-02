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
print(f"rc {rc}  2rc {2*rc}  L {L}  |F_0| {np.linalg.norm(F0):.3e}")
print(f"{'shell (A)':>12} {'n atoms':>8} {'sigma':>6} {'rms |dF_0|':>11} {'|dF|/|F|':>9} {'rms dE_0':>9}")
shells = [(lo, lo + 1.0) for lo in np.arange(0.0, 2 * rc, 1.0)] + [(0.9 * 2 * rc, 2 * rc), (2 * rc, 2 * rc + 1)]
for lo, hi in shells:
    sel = np.where((d_all >= lo) & (d_all < hi))[0]
    sel = sel[sel != 0]
    sigmas = [0.01, 0.05, 0.1] if lo >= 0.9 * 2 * rc - 1e-9 else [0.05]
    for s in sigmas:
        dE, dF = [], []
        for seed in range(10):
            b = a.copy()
            b.positions[sel] += np.random.default_rng(seed).normal(0, s, (len(sel), 3))
            E, F = ev(b)
            dE.append(E - E0); dF.append(np.linalg.norm(F - F0))
        rF = np.sqrt(np.mean(np.square(dF)))
        print(f"{lo:5.1f}-{hi:5.1f} {len(sel):8d} {s:6.2f} {rF:11.2e} {rF / np.linalg.norm(F0):9.2e} {np.sqrt(np.mean(np.square(dE))):9.2e}")

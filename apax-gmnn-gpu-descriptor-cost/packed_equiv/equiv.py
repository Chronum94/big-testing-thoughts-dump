"""Packed vs unpacked moments on trained checkpoints (CPU, no retraining): same params, same
structures, diff of energies and forces. Also val MAE under both.

usage (apax container, CPU): python equiv.py models/RUN [n_structures]
"""
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/home/chronum/scratch/semicond_dataset/analysis")
import cpuload  # noqa: E402,F401  (orbax GPU->CPU restore workaround; patches apax.train.checkpoints.load_params)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from ase.io import read  # noqa: E402
from vesin import NeighborList  # noqa: E402

import apax.train.checkpoints as ck  # noqa: E402
from apax.data.preprocessing import sort_by_centre  # noqa: E402
from apax.nn.builder import GMNNBuilder  # noqa: E402

run, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50
cfg = yaml.safe_load(open(f"{run}/config.yaml"))
params = ck.load_params(run, best=True)
frames = read(cfg["data"]["val_data_path"], f":{n}")
r_max = cfg["model"]["basis"]["r_max"]


def inputs(atoms):
    box = atoms.cell.array
    i, j, S = NeighborList(cutoff=r_max, full_list=True).compute(
        points=atoms.positions, box=box, periodic=True, quantities="ijS")
    idx, offsets = sort_by_centre(np.array([i, j], dtype=np.int32), S @ box)
    frac = atoms.positions @ np.linalg.inv(box).T  # apax convention (ase_calc: space.transform(inv(cell), R))
    return (jnp.asarray(frac), jnp.asarray(atoms.numbers), jnp.asarray(idx), jnp.asarray(box), jnp.asarray(offsets))


out = {}
for packed in (False, True):
    b = GMNNBuilder({**cfg["model"], "packed_moments": packed})
    E, F = [], []
    for at in frames:
        m = b.build_energy_derivative_model(apply_mask=True, init_box=np.asarray(at.cell.array), calc_stress=False)
        r = jax.jit(m.apply)(params, *inputs(at))
        E.append(float(r["energy"])); F.append(np.asarray(r["forces"]))
    out[packed] = (np.array(E), F)

(E0, F0), (E1, F1) = out[False], out[True]
nat = np.array([len(a) for a in frames])
dF = max(np.abs(a - b).max() for a, b in zip(F0, F1))
Fscale = max(np.abs(a).max() for a in F0)
Eref = np.array([a.get_potential_energy() for a in frames])
Fref = [a.get_forces() for a in frames]
mae = lambda E, F: (np.mean(np.abs(E - Eref) / nat) * 1e3, np.mean(np.concatenate([np.abs(f - g).ravel() for f, g in zip(F, Fref)])) * 1e3)
print(f"{run}: {len(frames)} structures, dtype {params and jax.tree.leaves(params)[0].dtype}")
print(f"  max |dE|/atom {np.max(np.abs(E0 - E1) / nat):.2e} eV   max |dF| {dF:.2e} eV/A (max |F| {Fscale:.1f})")
for k, (E, F) in (("unpacked", out[False]), ("packed", out[True])):
    e, f = mae(E, F)
    print(f"  {k:8} val MAE  E {e:.4f} meV/atom  F {f:.3f} meV/A")

"""Phase 0 of the radial-vs-angular study: cost map of GMNN descriptor settings on the GPU.

For each (A = n_radial, B = n_radial_tensor, drop_contractions) setting, built from the
emim_r55 model config (CP radial, r_max 5.5, nn 128x128):
  - features, parameters used by the system's elements, analytic scattered floats per pair
  - MD-like call: energy + forces (jitted model.apply), 480-atom EMIM cell and 3x3x3 box
  - train-like step: grad w.r.t. params of an energy + force MSE loss, 480-atom cell
Times are medians over repeated calls after compilation (block_until_ready).
Note (study plan): A/B measured, FLOPs not; wall time is the primary cost axis.

usage (GPU container): python costmap.py out.jsonl ['[[A, B, [drops], packed?], ...]']  (optional grid override)
"""

import json
import sys
import time

import jax

jax.devices()  # ROCm container: initialise the backend before TensorFlow loads
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from ase.io import read  # noqa: E402
from vesin import NeighborList  # noqa: E402

from apax.config.model_config import GMNNConfig  # noqa: E402
from apax.data.preprocessing import sort_by_centre  # noqa: E402
from apax.nn.builder import GMNNBuilder  # noqa: E402

BASE = yaml.safe_load(open("/home/chronum/scratch/apax_paper/runs/emim_r55.yaml"))["model"]
CELL = read("/home/chronum/scratch/apax_paper/data/emim_train.extxyz", 0)

GRID = [(a, b, []) for a in (8, 12, 16, 24, 32) for b in (2, 3, 4, 6, 8)]
GRID += [(12, 4, [4, 6]), (12, 4, [4, 6, 7]), (12, 4, [3, 6, 7]),  # L1, L2, L3
         (24, 4, [3, 6, 7]), (32, 4, [3, 6, 7]), (24, 4, [4, 6]), (12, 6, [3, 6, 7])]


def inputs(atoms, r_max):
    box = atoms.cell.array
    i, j, S = NeighborList(cutoff=r_max, full_list=True).compute(
        points=atoms.positions, box=box, periodic=True, quantities="ijS")
    idx, offsets = sort_by_centre(np.array([i, j], dtype=np.int32), S @ box)
    frac = np.linalg.solve(box.T, atoms.positions.T).T
    return (jnp.asarray(frac), jnp.asarray(atoms.numbers), jnp.asarray(idx),
            jnp.asarray(box), jnp.asarray(offsets)), len(i) / len(atoms)


def timeit(fn, *args, min_time=1.0):
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    n, t0 = 0, time.perf_counter()
    times = []
    while time.perf_counter() - t0 < min_time or n < 5:
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t)
        n += 1
    return 1000 * float(np.median(times))


def main(out, grid=None):
    tag = {} if grid is None else {"batch": time.strftime("%F %T")}
    grid = GRID if grid is None else [tuple(g) + (False,) * (4 - len(g)) for g in json.loads(grid)]
    systems = {"cell480": CELL, "box12960": CELL.repeat(3)}
    for a, b, drop, *packed in grid:
        packed = bool(packed and packed[0])
        cfg = {**BASE, "n_radial": a, "n_radial_tensor": b, "drop_contractions": drop, "packed_moments": packed}
        mc = GMNNConfig(**cfg).model_dump()
        builder = GMNNBuilder(mc)
        # unpacked moments: m1 3 + m2 9 + m3 27 floats per B channel; l_max 2 (drop c3) removes m3
        per_b = (9 if packed else 12) if 3 in drop else (19 if packed else 39)
        rec = {**tag, "A": a, "B": b, "drop": drop, "packed": packed, "floats_per_pair": a + per_b * b}
        for name, atoms in systems.items():
            x, ppa = inputs(atoms, mc["basis"]["r_max"])
            model = builder.build_energy_derivative_model(apply_mask=True, init_box=np.asarray(atoms.cell.array),
                                                          calc_stress=False)
            if name == "cell480":  # params do not depend on the system; jit avoids op-by-op compiles
                params = jax.jit(model.init)(jax.random.PRNGKey(0), *x)
            md = jax.jit(model.apply)
            rec[f"md_ms_{name}"] = timeit(md, params, *x)
            rec[f"pairs_per_atom_{name}"] = ppa
            if name == "cell480":
                def loss(p):
                    r = model.apply(p, *x)
                    return r["energy"] ** 2 + 4.0 * jnp.mean(r["forces"] ** 2)
                rec["train_ms_cell480"] = timeit(jax.jit(jax.grad(loss)), params)
                leaves = jax.tree_util.tree_leaves_with_path(params)
                nz = len(set(atoms.numbers))
                used = 0
                for path, v in leaves:
                    s = v.shape
                    if len(s) >= 2 and s[0] == 119 and s[1] == 119:
                        used += nz * nz * int(np.prod(s[2:]))
                    elif len(s) >= 1 and s[0] == 119:
                        used += nz * int(np.prod(s[1:]))
                    else:
                        used += v.size
                rec["params_used"] = int(used)
        print(json.dumps(rec), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])

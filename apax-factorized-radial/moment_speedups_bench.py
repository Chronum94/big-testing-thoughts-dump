# E+F(+stress) timing: sparse vs rows, fp64 vs fp32 distances. Usage: python bench.py [repeat]
import sys, time
import numpy as np, jax, jax.numpy as jnp
from ase.build import bulk
import apax
from apax.config.model_config import GMNNConfig
from apax.data.preprocessing import compute_nl

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # NaCl cubic repeat; 10 -> 8000 atoms
a = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(n); a.rattle(0.1, seed=0)
box = np.asarray(a.cell.array); frac = a.get_scaled_positions()
idx, off = compute_nl(frac, box, 5.0)
K = int(np.bincount(idx[1]).max())
print(f"N {len(a)} P {idx.shape[1]} K {K} backend {jax.default_backend()}")
args = (jnp.asarray(frac), jnp.asarray(a.numbers), jnp.asarray(idx, jnp.int32), jnp.asarray(box), jnp.asarray(off))
variants = {
    "sparse fp64": {},
    "sparse fp32": {"distance_dtype": "fp32"},
    "rows fp64": {"nbr_rows": K},
    "rows fp32": {"nbr_rows": K, "distance_dtype": "fp32"},
}
for stress in [False, True]:
    for name, kw in variants.items():
        cfg = GMNNConfig.model_validate({"name": "gmnn", "calc_stress": stress, **kw}).model_dump()
        m = GMNNConfig.model_validate(cfg).get_builder()(cfg).build_energy_derivative_model(init_box=box)
        params = m.init(jax.random.PRNGKey(0), *args)
        f = jax.jit(lambda p, *x: m.apply(p, *x))
        jax.block_until_ready(f(params, *args))
        ts = []
        for _ in range(50):
            t = time.perf_counter(); jax.block_until_ready(f(params, *args)); ts.append(time.perf_counter() - t)
        print(f"{'E+F+S' if stress else 'E+F':6s} {name:12s} median {1e3*np.median(ts):.3f} ms  min {1e3*np.min(ts):.3f} ms")

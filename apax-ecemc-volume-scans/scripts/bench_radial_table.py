"""Exact vs tabulated radial (Hermite / Chebyshev) on a trained EC:EMC model: force agreement and time per force call.

usage: python bench_radial_table.py MODEL_DIR [REPEAT] [N_TIMED]   (apax from ~/apax-wt/radial-tab on PYTHONPATH)
System: the first 2x-box test frame (286 atoms) tiled REPEAT^3 times. Times jitted energy+forces, after warm-up.
"""
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/chronum/scratch/common"); import cpuload  # noqa: E401,E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from ase.io import read  # noqa: E402

from apax.config.model_config import GMNNConfig  # noqa: E402
from apax.data.preprocessing import compute_nl  # noqa: E402
from apax.layers.descriptor.radial_table import tabulate_radial  # noqa: E402
from apax.utils.convert import atoms_to_inputs  # noqa: E402

model_dir = Path(sys.argv[1]); rep = int(sys.argv[2]) if len(sys.argv) > 2 else 3; n_timed = int(sys.argv[3]) if len(sys.argv) > 3 else 5
raw = yaml.safe_load(open(model_dir / "config.yaml"))["model"]
raw.pop("zero_init_output", None)  # init-only option from the training worktree; no effect on a trained model
cfg = GMNNConfig(**raw); params = cpuload.load_params(model_dir)  # loaded onto the CPU device
cpu, dev = jax.devices("cpu")[0], jax.devices()[0]  # tabulate on CPU (eager GPU ops hit an LLVM abort on ROCm/WSL), time on dev
r_max = cfg.basis.r_max

atoms = read("/home/chronum/scratch/ecemc/data/test_x2.extxyz", 0).repeat(rep)
inp = atoms_to_inputs([atoms])
R, Z, box = np.asarray(inp["positions"][0]), np.asarray(inp["numbers"][0]), np.asarray(inp["box"][0])
idx, off = compute_nl(R, box, r_max)
elements = sorted(set(atoms.numbers.tolist()))
print(f"{len(atoms)} atoms, {idx.shape[1]} pairs, elements {elements}, r_max {r_max}", flush=True)

cfg64 = cfg.model_dump(); cfg64["descriptor_dtype"] = "fp64"
exact64 = cfg.get_builder()(cfg64).build_radial_function()
radial_params = params["params"]["energy_model"]["representation"]["radial_fn"]
from apax.layers.descriptor.basis_functions import BesselBasis  # noqa: E402
rec_cfg = cfg.model_dump(); rec_cfg["basis"]["sine_recurrence"] = True
rec_basis = BesselBasis(n_basis=cfg.basis.n_basis, r_max=r_max, sine_recurrence=True, dtype=cfg.descriptor_dtype)
variants = {"exact": (None, cfg.model_dump()), "exact_rec": (None, rec_cfg)}  # name: (injected radial, model config)
with jax.default_device(cpu):
    variants["pair_table"] = (tabulate_radial(exact64, radial_params, elements, kind="pair_table", dtype=jnp.float32,
                                              basis_fn=cfg.get_builder()(cfg.model_dump()).build_basis_function()), cfg.model_dump())
    variants["pair_table_rec"] = (tabulate_radial(exact64, radial_params, elements, kind="pair_table", dtype=jnp.float32,
                                                  basis_fn=rec_basis), cfg.model_dump())
    for kind, n, deg in (("hermite", 256, 0), ("piecewise", 16, 7), ("piecewise", 32, 5), ("chebyshev", 32, 0)):
        name = f"{kind}{n}" + (f"_p{deg}" if kind == "piecewise" else "")
        variants[name] = (tabulate_radial(exact64, radial_params, elements, kind=kind, n=n, dtype=jnp.float32, degree=deg or 7),
                          cfg.model_dump())
params, R, Z, idx, box, off = jax.device_put((params, R, Z, idx, box, off), dev)
print("timing on", dev, flush=True)

ref, t_ref = None, None
for name, (radial, mcfg) in variants.items():
    model = cfg.get_builder()(mcfg, radial_fn=radial).build_energy_derivative_model(init_box=np.asarray(box))
    f = jax.jit(lambda p, r, z, i, b, o: model.apply(p, r, z, i, b, o))  # inputs as arguments, not baked-in constants
    t0 = time.perf_counter(); out = jax.block_until_ready(f(params, R, Z, idx, box, off)); t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n_timed):
        out = jax.block_until_ready(f(params, R, Z, idx, box, off))
    t = (time.perf_counter() - t0) / n_timed
    F, E = np.asarray(out["forces"]), float(out["energy"])
    if ref is None:
        ref, e_ref, t_ref = F, E, t
    dF = np.abs(F - ref); rms = np.sqrt((ref ** 2).mean())
    print(f"{name:12s} {t * 1e3:8.1f} ms/call ({t_ref / t:4.2f}x)  compile {t_compile:5.1f} s | "
          f"|dF| max {dF.max() * 1e3:.2e} meV/A  rms {np.sqrt((dF ** 2).mean()) * 1e3:.2e} meV/A (F rms {rms:.2f} eV/A) | "
          f"dE {abs(E - e_ref) * 1e3 / len(atoms):.2e} meV/atom", flush=True)

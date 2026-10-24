"""Piecewise Hermite (radial_table.py backend) on the trained model: worst relative error over 36 pairs x 17 channels,
table evaluated in float64 and in float32 (Horner precision). usage: python piecewise_check.py MODEL_DIR"""
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/chronum/scratch/common"); import cpuload  # noqa: E401,E402
import jax.numpy as jnp  # noqa: E402
import yaml  # noqa: E402
from apax.config.model_config import GMNNConfig  # noqa: E402
from apax.layers.descriptor.radial_table import tabulate_radial  # noqa: E402
raw = yaml.safe_load(open(f"{sys.argv[1]}/config.yaml"))["model"]; raw.pop("zero_init_output", None)
mc = GMNNConfig(**raw).model_dump(); mc["descriptor_dtype"] = "fp64"
radial = GMNNConfig(**raw).get_builder()(mc).build_radial_function()
rep = cpuload.load_params(sys.argv[1])["params"]["energy_model"]["representation"]
W = np.asarray(rep["radial_compression"]["kernel"]); r_min, r_max = 0.5, radial.basis_fn.r_max
ELEM = [1, 3, 6, 8, 9, 15]; pairs = [(c, n) for c in ELEM for n in ELEM]


def R_and_dR(d, c, n):
    f = lambda x: radial.apply({"params": rep["radial_fn"]}, x, jnp.full(x.shape, n), jnp.full(x.shape, c))  # noqa: E731
    R, dR = jax.jvp(f, (d,), (jnp.ones_like(d),)); R, dR = np.asarray(R), np.asarray(dR)
    return np.concatenate([R, R @ W], 1), np.concatenate([dR, dR @ W], 1)
fine = np.linspace(r_min, r_max, 20001)
ref = {(c, n): R_and_dR(jnp.asarray(fine), c, n) for c, n in pairs}
for deg, K in ((5, 16), (5, 32), (5, 64), (7, 8), (7, 16), (7, 32), (9, 8), (9, 16)):
    row = []
    for dt in (jnp.float64, jnp.float32):
        tab = tabulate_radial(radial, rep["radial_fn"], ELEM, kind="piecewise", n=K, degree=deg, dtype=dt)
        eR = edR = 0.0
        for (c, n), (Rf, dRf) in ref.items():
            d = jnp.asarray(fine, dt); zc, zn = jnp.full(d.shape, c), jnp.full(d.shape, n)
            f = lambda x: tab.apply({}, x, zn, zc)  # noqa: E731
            P, dP = jax.jvp(f, (d,), (jnp.ones_like(d),)); P = np.asarray(P, np.float64); dP = np.asarray(dP, np.float64)
            P = np.concatenate([P, P @ W], 1); dP = np.concatenate([dP, dP @ W], 1)
            eR = max(eR, (np.abs(P - Rf) / (np.abs(Rf).max(0) + 1e-30)).max()); edR = max(edR, (np.abs(dP - dRf) / (np.abs(dRf).max(0) + 1e-30)).max())
        row += [eR, edR]
    print(f"degree {deg} x {K:3d} intervals: fp64 R {row[0]:.1e} dR {row[1]:.1e} | fp32 R {row[2]:.1e} dR {row[3]:.1e} | "
          f"table {36 * K * (deg + 1) * 12 * 4 / 1024:.0f} KB (12 radial ch, fp32)", flush=True)

"""Active-learning sampling: Langevin MD with an apax student (ASE calculator) on small training-size cells.

usage: python student_ase_md.py MODEL_DIR STRUCT.extxyz IDX T_K PS DT_FS EVERY_FS OUT.extxyz [--tag t] [--skip_ps 0.5]
Writes frames (student energy/forces as student_energy / student_forces) every EVERY_FS after SKIP_PS.
Stops early (keeping frames so far) if any pair distance drops below 0.5 A or T exceeds 5x target.
"""
import argparse
import time

import jax

jax.devices()
import tensorflow  # noqa: E402,F401  (ROCm container: after the JAX backend, before apax)
import os  # noqa: E402
import sys  # noqa: E402

if os.environ.get("JAX_PLATFORMS") == "cpu":  # GPU-saved checkpoints need the CPU restore shim
    sys.path.insert(0, "/home/chronum/scratch/common")
    import cpuload  # noqa: F401
import numpy as np  # noqa: E402
from apax.md import ASECalculator  # noqa: E402
from ase import units  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("model", "struct", "idx", "T", "ps", "dt", "every", "out"):
    ap.add_argument(k)
ap.add_argument("--tag", default=""); ap.add_argument("--skip_ps", type=float, default=0.5)
a_ = ap.parse_args()
a = read(a_.struct, int(a_.idx)); a.calc = None; T, dt = float(a_.T), float(a_.dt)
a.calc = ASECalculator(a_.model)
MaxwellBoltzmannDistribution(a, temperature_K=T, rng=np.random.default_rng(int(T) + 1))
dyn = Langevin(a, dt * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=np.random.default_rng(1))
every, skip = max(1, round(float(a_.every) / dt)), round(a_.skip_ps * 1000 / dt)
out, t0, status = [], time.time(), "ok"


class Blowup(Exception):
    pass


def grab():
    d = a.get_all_distances(mic=True); np.fill_diagonal(d, 9)
    if d.min() < 0.5 or a.get_temperature() > 5 * T:
        raise Blowup(f"min d {d.min():.2f}, T {a.get_temperature():.0f} at {dyn.nsteps * dt / 1000:.2f} ps")
    if dyn.nsteps < skip:
        return
    b = a.copy(); b.calc = None
    b.arrays["student_forces"] = a.get_forces()
    b.info = {"category": f"al_{a_.tag}_{int(T)}K", "student_energy": float(a.get_potential_energy()), "T_target": T,
              "t_ps": dyn.nsteps * dt / 1000, "min_d": float(d.min())}
    out.append(b)


dyn.attach(grab, interval=every)
try:
    dyn.run(int(float(a_.ps) * 1000 / dt))
except Blowup as e:
    status = f"BLOWUP {e}"
write(a_.out, out)
print(f"{a_.tag} {T} K: {status}; {len(out)} frames, {dyn.nsteps} steps, {time.time() - t0:.0f}s ({(time.time() - t0) / max(1, dyn.nsteps) * 1000:.1f} ms/step)", flush=True)

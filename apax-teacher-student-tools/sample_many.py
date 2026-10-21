"""Many short student Langevin runs in one process (one container start, one compile per cell size).

usage: python sample_many.py MODEL_DIR STRUCTS.extxyz OUTDIR SPEC [SPEC ...]   SPEC = idx:T_K
  [--ps 1.5] [--dt 0.3] [--every 50] [--skip_ps 0.3]
Each run writes OUTDIR/s_<idx>_<T>.extxyz (student energy/forces as student_energy / student_forces) and stops early,
keeping its frames, if a pair distance drops below 0.5 A or T exceeds 5x target.
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
ap.add_argument("model"); ap.add_argument("structs"); ap.add_argument("outdir"); ap.add_argument("specs", nargs="+")
ap.add_argument("--ps", type=float, default=1.5); ap.add_argument("--dt", type=float, default=0.3)
ap.add_argument("--every", type=float, default=50); ap.add_argument("--skip_ps", type=float, default=0.3)
a_ = ap.parse_args()
calc = ASECalculator(a_.model)
starts = read(a_.structs, ":")


class Blowup(Exception):
    pass


for spec in a_.specs:
    idx, T = int(spec.split(":")[0]), float(spec.split(":")[1])
    if os.path.exists(f"{a_.outdir}/s_{idx}_{int(T)}.extxyz") or os.path.exists(f"{a_.outdir}/s_{idx}_{int(T)}.empty"):
        continue  # done before a crash/retry
    a = starts[idx].copy(); a.calc = calc; t0 = time.time(); out, status = [], "ok"
    MaxwellBoltzmannDistribution(a, temperature_K=T, rng=np.random.default_rng(idx * 7919 + int(T)))
    dyn = Langevin(a, a_.dt * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=np.random.default_rng(idx))
    every, skip = max(1, round(a_.every / a_.dt)), round(a_.skip_ps * 1000 / a_.dt)

    def grab():
        d = a.get_all_distances(mic=True); np.fill_diagonal(d, 9)
        if d.min() < 0.5 or a.get_temperature() > 5 * T:
            raise Blowup(f"min d {d.min():.2f}, T {a.get_temperature():.0f} at {dyn.nsteps * a_.dt / 1000:.2f} ps")
        if dyn.nsteps < skip:
            return
        b = a.copy(); b.calc = None
        b.arrays["student_forces"] = a.get_forces()
        b.info = {"category": f"al_{starts[idx].info.get('category', '?')}_{int(T)}K", "student_energy": float(a.get_potential_energy()),
                  "T_target": T, "t_ps": dyn.nsteps * a_.dt / 1000, "min_d": float(d.min()), "start_idx": idx}
        out.append(b)

    dyn.attach(grab, interval=every)
    try:
        dyn.run(int(a_.ps * 1000 / a_.dt))
    except Blowup as e:
        status = f"BLOWUP {e}"
    if out:
        write(f"{a_.outdir}/s_{idx}_{int(T)}.extxyz", out)
    else:
        open(f"{a_.outdir}/s_{idx}_{int(T)}.empty", "w").close()
    print(f"{spec} {starts[idx].info.get('category')} {len(a)} at: {status}; {len(out)} frames, {dyn.nsteps} steps, "
          f"{time.time() - t0:.0f}s ({(time.time() - t0) / max(1, dyn.nsteps) * 1000:.1f} ms/step)", flush=True)

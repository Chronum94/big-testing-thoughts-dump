"""Teacher (PET-MAD) Langevin NVT for held-out MD test frames.

usage: python teacher_md.py STRUCT.extxyz IDX T_K PS DT_FS EVERY_FS OUT.extxyz [--model m] [--skip_ps 1] [--tag name]
Frames after --skip_ps are written every EVERY_FS with teacher energy/forces/stress; info: category=test_md_<tag>_<T>K.
"""
import argparse
import time

import jax

jax.devices()
import numpy as np  # noqa: E402
from ase import units  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from petjax import UPETCalculator  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("struct", "idx", "T", "ps", "dt", "every", "out"):
    ap.add_argument(k)
ap.add_argument("--model", default="m"); ap.add_argument("--skip_ps", type=float, default=1.0); ap.add_argument("--tag", default="")
a_ = ap.parse_args()
a = read(a_.struct, int(a_.idx)); T, dt = float(a_.T), float(a_.dt)
a.calc = UPETCalculator.from_checkpoint(f"/home/chronum/scratch/petjax/checkpoints/pet-mad-{a_.model}", stress=True, bucket_strategy="powers_of_2")
MaxwellBoltzmannDistribution(a, temperature_K=T, rng=np.random.default_rng(int(T)))
dyn = Langevin(a, dt * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=np.random.default_rng(0))
every, skip = max(1, round(float(a_.every) / dt)), round(a_.skip_ps * 1000 / dt)
out, t0 = [], time.time()
info0 = {k: v for k, v in a.info.items() if isinstance(v, (int, float, str))}


def grab():
    if dyn.nsteps < skip:
        return
    b = a.copy()
    b.calc = SinglePointCalculator(b, energy=a.get_potential_energy(), forces=a.get_forces(), stress=a.get_stress())
    b.info = {**info0, "category": f"test_md_{a_.tag}_{int(T)}K", "T": T, "t_ps": dyn.nsteps * dt / 1000}
    out.append(b)


dyn.attach(grab, interval=every)
try:
    dyn.run(int(float(a_.ps) * 1000 / dt))
finally:
    write(a_.out, out)
    d = a.get_all_distances(mic=True); np.fill_diagonal(d, 9)
    Ts = [x.get_temperature() for x in out] if out and out[0].get_momenta() is not None else []
    print(f"{a_.struct}[{a_.idx}] {T} K: {len(out)} frames, {dyn.nsteps} steps, {time.time() - t0:.0f}s, "
          f"T mean {np.mean(Ts) if Ts else float('nan'):.0f} K, min d {d.min():.2f}", flush=True)

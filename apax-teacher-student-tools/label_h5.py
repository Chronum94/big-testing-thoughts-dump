"""Label frames sampled from apax MD trajectories (h5md) with a PET-MAD teacher (GPU).

usage: python label_h5.py OUT.extxyz EVERY_PS SKIP_PS RUN_DIR[:TAG] [RUN_DIR[:TAG] ...]
Frames carry info: category=al_<tag>, src=<run dir>, frame index; student energy/forces are kept as arrays/info
(student_energy, student_forces) so errors can be computed without rerunning the student.
"""
import sys

import jax

jax.devices()
import h5py  # noqa: E402
import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import write  # noqa: E402
from petjax import UPETCalculator  # noqa: E402

out_path, every_ps, skip_ps = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
calc = UPETCalculator.from_checkpoint("/home/chronum/scratch/petjax/checkpoints/pet-mad-m", stress=True, bucket_strategy="powers_of_2")
out = []
for spec in sys.argv[4:]:
    d, tag = (spec.split(":") + [None])[:2]
    try:
        f = h5py.File(f"{d}/md.h5", "r")
    except OSError as e:
        print("skip", d, e, flush=True); continue
    p, o = f["particles/atoms"], f["observables/atoms"]; n = len(p["position/time"])
    t = p["position/time"][:]; t = t - t[0]
    dt_frame = (t[1] - t[0]) if n > 1 else 1.0  # apax time units are internal; use the yaml duration instead when available
    import yaml
    yml = d + ".yaml"
    try:
        cfg = yaml.safe_load(open(yml)); total_ps = cfg["duration"] / 1000.0
    except Exception:
        total_ps = n * 0.05
    step = max(1, int(round(every_ps / (total_ps / n)))); first = int(skip_ps / (total_ps / n))
    Z = p["species/value"][0].astype(int)
    for k in range(first, n, step):
        a = Atoms(numbers=Z, positions=p["position/value"][k], cell=p["box/edges/value"][k], pbc=True)
        es, fs = float(o["potential_energy/value"][k].ravel()[0]), np.array(p["force/value"][k])
        a.calc = calc
        e, fo, s = a.get_potential_energy(), a.get_forces(), a.get_stress()
        a.calc = SinglePointCalculator(a, energy=e, forces=fo, stress=s)
        a.arrays["student_forces"] = fs
        a.info.update(category=f"al_{tag or d.split('/')[-1]}", src=d, frame=k, t_ps=k * total_ps / n, student_energy=es)
        out.append(a)
    print(f"{d}: {n} frames, labelled {len(range(first, n, step))}", flush=True)
write(out_path, out)
print("total", len(out))

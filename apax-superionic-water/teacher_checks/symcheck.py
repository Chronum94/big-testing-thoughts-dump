"""Forces on perfect symmetric ice X (all forces must vanish by site symmetry). Tests float32 vs float64 and extra_neighbors."""
import sys, numpy as np
from ase import Atoms
from petjax import UPETCalculator
ck = sys.argv[1]
def iceX(a):
    O = np.array([[0,0,0],[.5,.5,.5]])*a; H = np.array([[.25,.25,.25],[.75,.75,.25],[.75,.25,.75],[.25,.75,.75]])*a
    return Atoms("O2H4", positions=np.vstack([O,H]), cell=[a]*3, pbc=True)
for kw in [dict(), dict(default_dtype="float64"), dict(extra_neighbors=32), dict(default_dtype="float64", extra_neighbors=32)]:
    c = UPETCalculator.from_checkpoint(f"/home/chronum/scratch/petjax/checkpoints/{ck}", stress=True, **kw)
    for a in [3.0, 2.6, 2.4]:
        for rep in [1, 2, 3]:
            x = iceX(a).repeat(rep); x.calc = c
            f = x.get_forces()
            print(kw, a, rep, "E/mol %.5f" % (x.get_potential_energy()/len(x)*3), "fmax %.4f" % np.abs(f).max(), "fO %.4f fH %.4f" % (np.abs(f[::1][np.array(x.numbers)==8]).max(), np.abs(f[np.array(x.numbers)==1]).max()), flush=True)

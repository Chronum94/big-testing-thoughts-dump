"""Rotational invariance of the (non-equivariant) PET teacher: rotate cell+positions rigidly,
compare E/atom and back-rotated forces. Structure: rattled bcc ice (16 H2O) at lattice constant a."""
import sys, numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation
from petjax import UPETCalculator
ck = sys.argv[1]
calc = UPETCalculator.from_checkpoint(f"/home/chronum/scratch/petjax/checkpoints/{ck}", stress=True)
def ice(a, delta):
    O = np.array([[0,0,0],[.5,.5,.5]])*a; Hs = np.array([[.25,.25,.25],[.75,.75,.25],[.75,.25,.75],[.25,.75,.75]])*a
    O1 = np.array([[0,0,0],[1,1,0],[1,0,1],[0,1,1]])*a; H=[]
    for k in range(4):
        t = O1[k] if k<2 else O[1]; u=t-Hs[k]; u/=np.linalg.norm(u); H.append(Hs[k]+delta*u)
    return Atoms("O2H4", positions=np.vstack([O,H]), cell=[a]*3, pbc=True)
for a, dl, rat in [(3.0, 0.3, 0.0), (3.0, 0.3, 0.15), (2.6, 0.0, 0.0), (2.6, 0.0, 0.15), (2.4, 0.0, 0.15)]:
    x0 = ice(a, dl).repeat(2); x0.rattle(rat, seed=3)
    Es, dF, Fs = [], [], []
    for s in range(12):
        R = Rotation.random(random_state=s).as_matrix() if s else np.eye(3)
        x = x0.copy(); x.set_cell(x0.cell.array @ R.T, scale_atoms=False); x.positions = x0.positions @ R.T
        x.calc = calc; Es.append(x.get_potential_energy()/len(x)); Fs.append(x.get_forces() @ R)  # back-rotated
    Fs = np.array(Fs); Fm = Fs.mean(0)
    print(f"a={a} rattle={rat}: E/atom spread (max-min) {1e3*(max(Es)-min(Es)):.2f} meV/atom; "
          f"force RMS dev from rot-mean {np.sqrt(((Fs-Fm)**2).mean()):.4f} eV/A; |F| RMS {np.sqrt((Fm**2).mean()):.3f}; "
          f"max |F_mean| {np.abs(Fm).max():.4f}", flush=True)

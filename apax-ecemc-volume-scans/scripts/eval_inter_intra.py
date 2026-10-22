"""Student vs teacher errors split into intermolecular (molecular translation + rotation) and intramolecular force parts
(Magdau et al., npj Comput Mater 2023 style), per molecule type, plus the volume-scan energy curve error.

usage: python eval_inter_intra.py MODEL_DIR FRAMES.extxyz [FRAMES ...]   (CPU; frames need arrays['mol_id'] + teacher E/F)
Per molecule M (atoms i, masses m_i, COM-relative positions r_i):
  translational  F_t,i = m_i / M * sum_j F_j
  rotational     F_r,i = m_i * (alpha x r_i),  alpha = I^-1 tau,  tau = sum_j r_j x F_j   (I = inertia tensor about the COM)
  intramolecular F_v,i = F_i - F_t,i - F_r,i
Errors: RMSE per component of (student - teacher) components, and RMSE / RMS(teacher component) ("rel"), per molecule type.
Volume scans (info scan_s): per-atom energy error (mean, MAE) for each scale factor s.
"""
import collections
import sys

sys.path.insert(0, "/home/chronum/scratch/common"); import cpuload  # noqa: E401,F401
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

from apax.md import ASECalculator  # noqa: E402

exec("def whole" + open("/home/chronum/scratch/ecemc/volume_scan.py").read().split("def whole")[1].split("out, skipped")[0])  # noqa: S102


def name_of(sym):
    c = collections.Counter(sym); key = "".join(f"{e}{c[e]}" for e in sorted(c))
    return {"C3H4O3": "EC", "C4H8O3": "EMC", "C3H6O3": "DMC", "F6P1": "PF6", "Li1": "Li", "F1Li1": "LiF", "F1H1": "HF"}.get(key, key)


def decompose(a, F, mol):
    P = whole(a, mol); m = a.get_masses(); out = {"trans": np.zeros_like(F), "rot": np.zeros_like(F), "intra": np.zeros_like(F)}
    for g in np.unique(mol):
        idx = mol == g; mm = m[idx]; M = mm.sum(); r = P[idx] - (mm[:, None] * P[idx]).sum(0) / M; f = F[idx]
        ft = mm[:, None] / M * f.sum(0)
        if idx.sum() > 1:
            I = (mm * (r * r).sum(1)).sum() * np.eye(3) - np.einsum("i,ia,ib->ab", mm, r, r)
            alpha = np.linalg.pinv(I) @ np.cross(r, f).sum(0); fr = mm[:, None] * np.cross(alpha, r)
        else:
            fr = np.zeros_like(f)
        out["trans"][idx], out["rot"][idx], out["intra"][idx] = ft, fr, f - ft - fr
    return out


calc = ASECalculator(sys.argv[1]); frames = [a for f in sys.argv[2:] for a in read(f, ":")]
pred = calc.batch_eval(frames, batch_size=8, silent=True)
err = collections.defaultdict(lambda: collections.defaultdict(list)); ref = collections.defaultdict(lambda: collections.defaultdict(list))
escan = collections.defaultdict(list); emd = []
for a, p in zip(frames, pred):
    mol = a.arrays["mol_id"]; sym = np.array(a.get_chemical_symbols())
    dT, dS = decompose(a, a.get_forces(), mol), decompose(a, p.get_forces(), mol)
    names = {g: name_of(sym[mol == g]) for g in np.unique(mol)}; atype = np.array([names[g] for g in mol])
    for comp in ("trans", "rot", "intra"):
        for t in set(atype):
            s = atype == t
            err[t][comp].append((dS[comp][s] - dT[comp][s]).ravel()); ref[t][comp].append(dT[comp][s].ravel())
            err["all"][comp].append((dS[comp][s] - dT[comp][s]).ravel()); ref["all"][comp].append(dT[comp][s].ravel())
    de = 1e3 * (p.get_potential_energy() - a.get_potential_energy()) / len(a)
    (escan[a.info["scan_s"]] if "scan_s" in a.info else emd).append(de)

print(f"model {sys.argv[1]}, {len(frames)} frames. Force-component RMSE in meV/A (rel = RMSE / RMS of the teacher component)")
print("| molecule | n atoms | trans RMSE (rel) | rot RMSE (rel) | intra RMSE (rel) | teacher RMS trans / rot / intra |")
print("|---|---|---|---|---|---|")
for t in ["all"] + sorted(k for k in err if k != "all"):
    row = []; rms = []
    for comp in ("trans", "rot", "intra"):
        e, r = np.concatenate(err[t][comp]), np.concatenate(ref[t][comp])
        rm = np.sqrt((r ** 2).mean()); rmse = np.sqrt((e ** 2).mean()); rms.append(1e3 * rm)
        row.append(f"{1e3 * rmse:.1f} ({rmse / rm:.2f})" if rm > 1e-9 else "–")
    print(f"| {t} | {len(np.concatenate(ref[t]['trans'])) // 3} | " + " | ".join(row) + " | " + " / ".join(f"{x:.0f}" for x in rms) + " |")
if emd:
    print(f"\nMD frames: per-atom E error mean {np.mean(emd):+.2f}, MAE {np.mean(np.abs(emd)):.2f} meV/atom (n={len(emd)})")
if escan:
    print("| scan s | n | E error mean (meV/atom) | MAE |\n|---|---|---|---|")
    for s in sorted(escan):
        print(f"| {s:.2f} | {len(escan[s])} | {np.mean(escan[s]):+.2f} | {np.mean(np.abs(escan[s])):.2f} |")

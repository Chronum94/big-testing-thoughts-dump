"""Cubic elastic constants by finite strain, relaxed-ion and clamped-ion, for one calculator.

usage: elastic.py teacher CKPT_DIR  OUT.jsonl [names...]
       elastic.py apax    MODEL_DIR OUT.jsonl [names...]
Per material: relax a (hydrostatic cell filter), then 6 Voigt strains x (+-0.5, +-1 %),
stress-strain slope -> C_ij (GPa); cubic-symmetrised C11/C12/C44, B, G_VRH, Born stability.
Resumable: skips names already in OUT.
"""
import json
import os
import sys

import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.units import GPa

CUBIC = {  # name: (structure, experimental a / A), same table as semicond_dataset/gen_seed.py
    "C": ("diamond", 3.567), "Si": ("diamond", 5.431), "Ge": ("diamond", 5.658),
    "Sn": ("diamond", 6.489), "SiC": ("zincblende", 4.360),
    "BN": ("zincblende", 3.615), "BP": ("zincblende", 4.538), "BAs": ("zincblende", 4.777),
    "AlP": ("zincblende", 5.463), "AlAs": ("zincblende", 5.661), "AlSb": ("zincblende", 6.136),
    "GaP": ("zincblende", 5.451), "GaAs": ("zincblende", 5.653), "GaSb": ("zincblende", 6.096),
    "InP": ("zincblende", 5.869), "InAs": ("zincblende", 6.058), "InSb": ("zincblende", 6.479),
    "ZnS": ("zincblende", 5.409), "ZnSe": ("zincblende", 5.668), "ZnTe": ("zincblende", 6.104),
    "CdTe": ("zincblende", 6.481), "HgTe": ("zincblende", 6.460), "CdSe": ("zincblende", 6.050),
    "PbS": ("rocksalt", 5.936), "PbSe": ("rocksalt", 6.124), "PbTe": ("rocksalt", 6.462),
    "SnTe": ("rocksalt", 6.320),
}
STRAINS = (-0.01, -0.005, 0.005, 0.01)
FMAX = 1e-4


def make_calc(kind, path):
    if kind == "teacher":
        from petjax import UPETCalculator
        return UPETCalculator.from_checkpoint(path, stress=True)
    import jax
    jax.devices()
    import tensorflow  # noqa: F401  (container LLVM clash: must precede apax)
    import apax.train.checkpoints as ck
    from apax.md import ASECalculator

    def load_params(model_version_path, best=True):
        # ponytail: GPU-saved orbax checkpoints won't restore on CPU without explicit shardings;
        # also takes the final (EMA) checkpoint `latest/`, not min-val-loss `best/`.
        import glob
        import orbax.checkpoint as ocp
        p = sorted(glob.glob(f"{model_version_path}/latest/*/default"), key=lambda s: int(s.split("/")[-2]))[-1]
        dev = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        pc = ocp.PyTreeCheckpointer()
        meta = pc.metadata(p)
        meta = getattr(getattr(meta, "item_metadata", meta), "tree", getattr(meta, "item_metadata", meta))
        st = pc.restore(p, restore_args=jax.tree.map(lambda m: ocp.ArrayRestoreArgs(sharding=dev), meta))
        return jax.tree.map(jax.numpy.asarray, st["model"]["params"])

    ck.load_params = load_params
    return ASECalculator(path)


def strained(a0, j, e):
    eps = np.zeros((3, 3))
    if j < 3:
        eps[j, j] = e
    else:  # Voigt 4,5,6 = yz, xz, xy; engineering shear strain e -> tensor e/2
        p, q = {3: (1, 2), 4: (0, 2), 5: (0, 1)}[j]
        eps[p, q] = eps[q, p] = e / 2
    a = a0.copy()
    a.set_cell(a0.cell.array @ (np.eye(3) + eps), scale_atoms=True)
    return a


def cij(name, calc):
    struct, a_exp = CUBIC[name]
    a0 = bulk(name, struct, a=a_exp, cubic=True)
    a0.calc = calc
    BFGS(FrechetCellFilter(a0, hydrostatic_strain=True), logfile=None).run(fmax=FMAX, steps=300)
    a_rel = float(a0.cell.lengths()[0])
    s0 = a0.get_stress()  # ASE: Voigt xx yy zz yz xz xy, eV/A^3, positive = tensile
    C = {"relaxed": np.zeros((6, 6)), "clamped": np.zeros((6, 6))}
    for j in range(6):
        ds = {"relaxed": [], "clamped": []}
        for e in STRAINS:
            a = strained(a0, j, e)
            a.calc = calc
            ds["clamped"].append(a.get_stress() - s0)
            BFGS(a, logfile=None).run(fmax=FMAX, steps=300)
            ds["relaxed"].append(a.get_stress() - s0)
        for k in C:
            C[k][:, j] = np.polyfit(STRAINS, np.array(ds[k]), 1)[0] / GPa
    out = dict(name=name, a=a_rel, a_exp=a_exp, s0_GPa=float(np.abs(s0).max() / GPa))
    for k, M in C.items():
        c11, c12, c44 = np.mean(np.diag(M)[:3]), np.mean(M[:3, :3][~np.eye(3, dtype=bool)]), np.mean(np.diag(M)[3:])
        B = (c11 + 2 * c12) / 3
        Gv, Gr = (c11 - c12 + 3 * c44) / 5, 5 * (c11 - c12) * c44 / (4 * c44 + 3 * (c11 - c12))
        out[k] = dict(C11=c11, C12=c12, C44=c44, B=B, G=(Gv + Gr) / 2,
                      stable=bool(c11 - c12 > 0 and c11 + 2 * c12 > 0 and c44 > 0),
                      asym=float(np.abs(M - M.T).max()), full=np.round(M, 2).tolist())
    return out


def selftest():
    """EMT Cu (fcc, cubic): clamped == relaxed, C symmetric, C11 > C12 > 0."""
    from ase.calculators.emt import EMT
    CUBIC["Cu"] = ("fcc", 3.61)
    r = cij("Cu", EMT())
    for k in ("relaxed", "clamped"):
        assert r[k]["stable"] and r[k]["asym"] < 1.0 and r[k]["C11"] > r[k]["C12"] > 0, r[k]
    assert abs(r["relaxed"]["C44"] - r["clamped"]["C44"]) < 0.5  # fcc: no internal relaxation
    print("selftest ok", {k: round(r["relaxed"][k], 1) for k in ("C11", "C12", "C44", "B")})


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
        sys.exit()
    kind, path, out = sys.argv[1:4]
    names = sys.argv[4:] or list(CUBIC)
    done = {json.loads(line)["name"] for line in open(out)} if os.path.exists(out) else set()
    calc = make_calc(kind, path)
    for n in names:
        if n in done:
            continue
        r = cij(n, calc)
        r["model"] = os.path.basename(path.rstrip("/"))
        print(n, f"a={r['a']:.4f}", {k: round(v, 1) for k, v in r["relaxed"].items() if k in ("C11", "C12", "C44", "B")}, flush=True)
        open(out, "a").write(json.dumps(r) + "\n")

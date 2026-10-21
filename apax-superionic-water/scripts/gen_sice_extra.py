import time
"""Superionic water v0b, following the literature agent's guidance (lit/SICE_PLAN.md): densities 2.2-4.3 g/cm3
(teacher credible ~15-315 GPa with PET-MAD-m), small H rattles (<= 0.08 A), O 0.1-0.15 A, max|F| <= 20 eV/A.
Relaxed (40 FIRE steps) superionic-like snapshots are the core, since random interstitial H is far from thermal.

usage: python gen_sice_extra.py out.extxyz
"""
import sys

src = open("/home/chronum/scratch/sice/gen_sice.py").read().split("t0 = time.time()")[0]
exec(src)  # functions, teacher, rng helpers from v0 (same file, our own code)
rng = np.random.default_rng(23)  # noqa: F821
frames.clear()  # noqa: F821
RHO = (2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3)
for rho in RHO:
    L = box_len(16, rho)  # noqa: F821
    for k in range(4):
        add(relaxed(ice_vii(L, False), 40), "V_iceVII_relaxed", rho, 0.08, 0.05)  # noqa: F821
    add(relaxed(ice_vii(L, True), 40), "V_iceX_relaxed", rho, 0.08, 0.05)  # noqa: F821
    for kind, nmol, nh in (("bcc", 3, 108), ("fcc", 2, 64)):
        if kind == "fcc" and rho < 3.1:
            continue
        L = box_len(54 if kind == "bcc" else 32, rho); o = lattice(kind, nmol); tet, octs = interstices(kind, nmol)  # noqa: F821
        cell = np.eye(3) * L  # noqa: F821

        def snap():
            h = pick_sites(np.vstack([tet, octs]) + rng.normal(0, 0.08 / L, (len(tet) + len(octs), 3)), nh, cell, 0.85)  # noqa: F821
            if h is None:
                return False
            a = relaxed(assemble(o, h, L), 40)  # noqa: F821
            return add(a, f"S_{kind}_relaxed40", rho, 0.1, 0.05) | add(a, f"S_{kind}_relaxed40", rho, 0.15, 0.08)  # noqa: F821
        n_times(6, snap)  # noqa: F821
    if rho <= 3.4:
        L = box_len(54, rho); o = lattice("bcc", 3); cell = np.eye(3) * L  # noqa: F821
        n_times(3, lambda: add(relaxed(assemble(o, molecular_h(o @ cell, cell), L), 40), "M_molecular_relaxed", rho, 0.08, 0.05))  # noqa: F821
    print(f"rho {rho}: {len(frames)} frames so far", flush=True)  # noqa: F821
print(f"generated {len(frames)} frames", flush=True)  # noqa: F821
write(sys.argv[1] + ".raw.extxyz", frames)  # noqa: F821
out, dropped = [], 0
t1 = time.time()  # noqa: F821
for i, a in enumerate(frames):  # noqa: F821
    if i % 20 == 0:
        print(f"label {i}/{len(frames)} n_atoms {len(a)} {time.time() - t1:.0f}s", flush=True)  # noqa: F821
    a.calc = calc  # noqa: F821
    e, f, s = a.get_potential_energy(), a.get_forces(), a.get_stress()
    if np.abs(f).max() > 20 or not np.isfinite(e):  # noqa: F821
        dropped += 1; continue
    a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=s)  # noqa: F821
    a.info["P_GPa"] = float(-s[:3].mean() * 160.2177); out.append(a)
write(sys.argv[1], out)  # noqa: F821
cats = [a.info["category"] for a in out]
print({c: cats.count(c) for c in dict.fromkeys(cats)}, f"dropped {dropped}, total {len(out)}")

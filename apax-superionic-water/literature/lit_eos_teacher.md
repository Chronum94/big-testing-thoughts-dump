# Ice VII/X EOS, H-bond symmetrisation, teacher (PET-MAD/UPET) coverage — literature extract (2026-10-20)

Verification key: **P-abs** = read in the primary abstract; **P-txt** = read in the primary full text (where given); **S(x)** = read in a secondary source x (primary paywalled); **D** = derived by me (eos_eval.py), not a published number; **U** = unverified.
Secondary sources: Fukui+ arXiv:2104.11107 Table 1 (=S-Fk); Frank+ 2004 GCA review section (=S-Fr); Sugimura+ 2010 PRB 82,134103 Table II/text (=S-Su10); Brown & Journaux 2020 Minerals 10,92 (=S-BJ). PDFs/text in /home/chronum/scratch/untrusted/sice_lit/eos_teacher/.
1 cm3/mol = 1.66054 Å3/H2O. bcc-O cell holds 2 H2O, a = (2 V_mol)^(1/3), d_OO = a·√3/2.

## Part 1. Static 300 K EOS of ice VII / X

| Study | Form | V0 (cm3/mol) | K0 (GPa) | K0' | P range (GPa) | DOI | Verified |
|---|---|---|---|---|---|---|---|
| Hemley+ 1987 Nature 330,737 | BM3 | 12.3(3) | 23.7(9) | 4.15(7) | 4.3–128 | 10.1038/330737a0 | S-Fr text, S-Fk T1 |
| Hemley data refit <70 GPa (by Frank) | BM3 | 12.2(1) | 24.8(1.2) | 4.14(12) | <70 | – | S-Fr text |
| Wolanin+ 1997 PRB 56,5781 (H2O; D2O to 60) | Vinet, K0 fixed (from Shimizu 1996) | 12.11(9) | 14.9(8) fixed | 6.2(1) | to 106 | 10.1103/PhysRevB.56.5781 | S-Fr text; IUCr HP datasheet gives 12.12(8)/14.9/6.2 |
| Wolanin 1997, BM variant | BM, K0 fixed | 12.37(9) | 14.9 | 5.4(1) | 5–103 | same | S-Fr, S-Fk T1 |
| Loubeyre+ 1999 Nature 397,503 (single crystal) | Vinet | 14.52 | 4.26 | 7.75 | 2.2–170 | 10.1038/17300 | S-Fr text, S-Fk T1 |
| Frank, Fei, Hu 2004 GCA 68,2781 | BM3 | 12.4(1) | 21.1(1.3) | 4.4(1) | 6.57–60.52 (300–900 K data) | 10.1016/j.gca.2003.12.007 | P-txt |
| Sugimura+ 2008 PRB 77,214103, ice VII part | Vinet, V0 fixed to Loubeyre | 14.52 | 5.02 | 7.51 | VII <40 GPa | 10.1103/PhysRevB.77.214103 | S-Su10 text |
| Sugimura 2008: compressibility kinks | – | – | – | – | changes at 40 and 60 GPa (VII→dd-VII→dd-X); data to 126 | same | P-abs |
| Somayazulu+ 2008 JCP 128,064510 | Vinet | 14.52 | 4.21 | 7.77 | 5–48 | – | S-Fk T1 |
| Klotz+ 2017 PRB 95,174111 (D2O, neutrons) | (per S-Fk) | 12.717 | 13.8 | 5.9 | 2.1–13.7 | 10.1103/PhysRevB.95.174111 | S-Fk T1; V0 12.7218 cm3/mol = 42.25 Å3/cell also in S-BJ |
| Fukui+ 2021 arXiv:2104.11107, 300 K | BM3 | 11.8(2) | 30.0(3.9) | 3.9(2) | 5–39 | arXiv | P-txt T1 |
| same, V0 fixed | BM3 | 12.717 | 17.7(8) | 4.7(2) | 5–39 | | P-txt T1 |
| Grande+ 2022 PRB 105,104109, cubic VII | Vinet | 42.50(88) Å3/cell (=12.80) | 18.47(4.00) | 2.51(1.51) | 2.7–5.1 | 10.1103/PhysRevB.105.104109 | P-txt (OSTI/arXiv:1906.11990 Fig 4 caption) |
| Grande 2022, tetragonal VII_t | Vinet | 41.11(53) Å3/cell (=12.38) | 20.76(2.46) | 4.49(35) | 5.1–30.9 | same | P-txt |
| Grande 2022, ice X | Vinet | 33.82(43) Å3/cell (=10.18) | 50.52(4.16) | 4.50(15) | >30.9, data to 88 | same | P-txt |
| Grande 2022, single-phase fit | Vinet | 43.05(20) Å3/cell (=12.96) | 12.57(50) | 6.06(7) | 2.7–88 | same | P-txt |
| Brown & Journaux 2020 (meta-fit of 14 expt + 2 DFT sets) | 4th-order Eulerian, V0 fixed (Klotz) | 12.7218 | 19.2 | 3.8 (K''=−0.09 GPa⁻¹), Prms 3.0 GPa | to 450 (>150 = DFT only); also b-spline LBF fits | 10.3390/min10020092 | P-txt T3 |
| Bezacier+ 2014 | – | – | – | – | <10 GPa (B&J Fig.2 legend) | – | U (params not found) |

Notes: published K0 scatter 4–24 GPa (Klotz 2017 per search snippet); V0/K0/K0' are strongly correlated, so compare V(P), not parameters. Brown & Journaux excluded Hemley (pressure calibration) and Loubeyre (deviate substantially) from their fit (P-txt).

### V(P) at selected pressures, 300 K
Measured-derived (Fukui 2021 Table 2, V0-fixed BM3, 300 K; P-txt): 10 GPa 9.55(9), 35 GPa 7.40(12), 60 GPa 6.49(13) cm3/mol.

Fits evaluated by me (**D**; extrapolated beyond each fit's range where marked). Format V cm3/mol / Å3 per H2O / d_OO Å:

| Fit | 50 GPa | 100 GPa | 150 GPa |
|---|---|---|---|
| Hemley BM3 | 6.756 / 11.22 / 2.443 | 5.568 / 9.25 / 2.290 | 4.929 / 8.18 / 2.199 |
| Loubeyre Vinet | 6.656 / 11.05 / 2.430 | 5.644 / 9.37 / 2.300 | 5.091 / 8.45 / 2.223 |
| Wolanin Vinet | 6.637 / 11.02 / 2.428 | 5.590 / 9.28 / 2.293 | 5.008 / 8.32 / 2.211 (extrap) |
| Frank BM3 | 6.755 / 11.22 / 2.442 | 5.610 / 9.32 / 2.296 (extrap) | 4.996 / 8.30 / 2.209 (extrap) |
| Grande ice X Vinet | 6.751 / 11.21 / 2.442 | 5.681 / 9.43 / 2.306 (extrap) | 5.057 / 8.40 / 2.218 (extrap) |
| Brown–Journaux BM4 | 6.722 / 11.16 / 2.439 | 5.681 / 9.43 / 2.306 | 5.145 / 8.54 / 2.231 |

Spread ~1.5% in V at 50 GPa, ~3% at 100 GPa, ~4% at 150 GPa.

### High-T EOS of ice VII
| Study | Result | Verified |
|---|---|---|
| Lai+ 2022/23 Am. Mineral. (doi 10.2138/am-2022-8554), single-crystal XRD to 78 GPa, 1000 K | HT-BM: K_T0 21.0(4) GPa, K' 4.45(6), dK/dT −0.009(4) GPa/K, α = 15(5)e−5 + 15(8)e−8·(T−300) K⁻¹; no volume discontinuity observed | P-abs |
| Sugimura+ 2010 PRB 82,134103, to ~50 GPa, 431–881 K | 300 K Vinet (14.52, 5.02, 7.51) + Anderson–Grüneisen: α0 = 150(19)e−5 K⁻¹, δT = 5.1(2) | P-txt (garbled in extraction; check before use) |
| Fei, Mao, Hemley 1993 JCP 99,5369 | V0 12.3(2), K0 23.9(7), K' 4.2(5), δT 6.0 (BM) | S-Su10 Table II |
| Frank 2004 | δT 4.8 | S-Su10 Table II |

## Part 2. H-bond symmetrisation (VII/VIII → X)

### Experiment
| Source | Value | Method | Verified |
|---|---|---|---|
| Goncharov+ 1996 Science 273,218 | 60 GPa (H2O), 70 GPa (D2O), 85–300 K | IR reflectance (ν3 soft mode); data to 210 GPa | P-abs |
| Aoki+ 1996 PRB 54,15673 | 62.1 GPa, RT | IR absorption to 110 GPa | P-abs |
| Loubeyre+ 1999 | "proton centring", XRD to 170 GPa; values not obtained | XRD | U (paywalled; Nature login) |
| Sugimura 2008 | dd-VII from 40 GPa, dd-X from 60 GPa, RT | XRD compressibility | P-abs |
| Meier+ 2018 **Nat. Commun.** 9, 2766 (not Nat. Phys.) | two transitions ~20 and ~75 GPa; tunnelling mode up to 97 GPa | 1H-NMR | P-abs |
| Meier (later NMR, per Komatsu) | max H mobility at d_OO 2.44–2.45 Å | NMR | S(Komatsu 2024) |
| Grande+ 2022 | X onset 30.9±3 GPa (laser-annealed), VII_t at 5.1 GPa | XRD+Raman | P-abs |
| Komatsu+ 2024 Nat. Commun. 15, 5100 (D2O) | bimodal→unimodal D at ~80 GPa; extrapolated d(O-D)=d(D…O) at ~120 GPa | neutron diffraction to 106 GPa | P-abs/P-txt |
| Brillouin (Zhang+ 2021 GRL) | softening 42–54 GPa; X at ~64 GPa | Brillouin, NaCl-doped ice | P-abs |
| Zha+ (via B&J T1) | p20 Raman mode at 90 GPa | Raman | S-BJ |

### Theory (classical vs NQE)
| Source | Classical | With NQE | Functional | Verified |
|---|---|---|---|---|
| Benoit, Marx, Parrinello 1998 Nature 392,258 | – | Pc 72 GPa (PIMD) | GGA (Becke 88; BLYP-type) | S(Zhang&Wang 2025 Tab I); d_OO U |
| Benoit, Romero, Marx 2002 PRL 89,145501 | centring ~110 GPa at 300 K (CP-MD) | – | U | P-abs |
| Kang+ 2013 Sci. Rep. 3,3272 | – | 61.6 GPa PIMD | PBE-D2 | S(Z&W) |
| Bronstein+ 2014 PRB 89,214101 | barrier vanishes ~100 GPa (group page) | ~65 GPa, QTB | PBE | P-abs (65); 100 S(INSP page) |
| Kuwahata & Tachikawa 2024 JCP 160,214501 | – | 61 GPa PIMD | revPBE-D3 | S(Z&W) |
| Cherubini+ 2024 PRB 110,014112 (arXiv:2403.09238) | Pc 116 GPa (0 K), 98 (50 K), 80 GPa (300 K) | ~51 GPa SSCHA (T-independent 0–300 K; ~1% ΔV); PIMD 43 GPa | PBE-trained NNP | P-txt; 43 S(Z&W) |
| Zhang & Wang 2025 arXiv:2507.01452 | ">100 GPa" | 60.6 GPa (NCT); symmetrisation at d_OO ≈ 2.38 Å; PBE Pc >10 GPa below SCAN | SCAN (validated vs QMC) | P-txt |
| Grande 2022 own DFT (0 K static) | ~90 GPa | – | U | P-txt |

Classical/static DFT puts symmetrisation at ~80–116 GPa (PBE); NQE moves it to ~50–72 GPa. The Zhang & Wang text gives d_OO 2.273 Å "≈30 GPa" and 2.520 Å "120 GPa", which runs the wrong way (d_OO falls as P rises); these look like typos, don't use them. From the EOS table above, d_OO ≈ 2.38–2.40 Å corresponds to ~60–65 GPa (D).

## Part 3. Teacher training-data coverage

| Item | Value | Source | Verified |
|---|---|---|---|
| MAD-1 / PET-MAD-1 functional | PBEsol, non-spin-polarised, QE+SIRIUS, SSSP 1.2 eff., 110/1320 Ry | arXiv:2506.19674 §IV B | P-txt |
| MAD-1 subsets | MC3D 33596; MC3D-rattled 30044 (Gaussian σ = 20% covalent radius); MC3D-random 2800 (85 el., isotropic volume to covalent-radii volume); surfaces 5589; clusters (2–8 at.) 9071; MC2D 2676; SHIFTML molcrys 8578; molfrags 3241 | 2506.19674 Table I, §IV A | P-txt |
| MD in MAD-1 | MC3D/MC2D/SHIFTML are "stable, or low-temperature MD configurations"; no explicit high-T MD subset | 2506.19674 §III A | P-txt |
| PET-MAD-1 cutoff | 4.5 Å, 2 MP layers | arXiv:2503.14118 Methods | P-txt |
| PET-MAD-1 water note | inherits GGA melting-point overestimate: RT water is undercooled and over-structured, too much proton delocalisation along H-bonds | 2503.14118 | P-txt |
| MAD-1.5/1.6 functional | r2SCAN, all-electron (FHI-aims); plus auxiliary PBE head (discarded) | arXiv:2603.02089 | P-txt |
| MAD-1.6 additions | monomers, dimers (0.125–1.0 × Σ vdW radii), trimers / trimers-ext (0.25–1.0 ×), MC3D-ext 11562, MC3D-random-ext 5397, binary-random bcc/fcc 73584 (vdW-volume cells), MAD-cat 53565 (adsorbates incl. H2O, OH, H2, O2 on metal slabs); total 362,646 structures | 2603.02089 Table I | P-txt |
| PET-MAD-1.6 cutoffs | XS/S/M: 7.5/8.0/8.5 Å, adaptive neighbours 16/24/32; fine-tuned from PET-OMat | 2603.02089 Table III; UPET docs | P-txt |
| Stability test | REMD 300–3000 K of a 102-element nanoparticle (a test, not training data) | 2603.02089 | P-txt |
| UPET docs limits | no stated limits on pressure, compression, water or temperature | lab-cosmo.github.io/upet models page | P-txt (WebFetch) |
| OMat24 (pretraining) | PBE/PBE+U VASP; rattled-Boltzmann (σ 0.5 Å, cell strain μ 1%, σ 5%, T_sample 300/500/1000 K); AIMD NVT/NPT 1000 and 3000 K, 50 steps, 2 fs; rattled relax (σ 0.2 Å, strain σ 4%); from Alexandria | arXiv:2410.12771 §4.1 | P-txt |
| OMat24 stress | wider max-stress distribution than MPtrj/Alex; training sAlex filter stress >80 GPa (Alex only) | 2410.12771 | P-txt; OMat24 max-stress values U (figure only) |
| uMLIPs under pressure (Loew+ 2025, arXiv:2508.17792; no PET-MAD tested) | energy MAE grows with P, e.g. eSEN-30M-OAM 4.1→41.7, MACE-MPA-0 6.9→163.5 meV/atom from 0→150 GPa; fine-tuning fixes it | Tables II–III | P-txt |
| PET-MAD / uMLIP benchmark on ice VII/X or superionic ice | none found | searches | – |

**Teacher-coverage summary.** Neither MAD generation contains any deliberately compressed or high-pressure subset. MC3D comes from experimental ambient crystals. The random-type subsets set cell volumes from covalent or vdW radii, i.e. near-ambient density. Rattling moves atoms without systematic volume reduction. Dimers and trimers sample short pair distances in vacuum only: down to 0.125 × Σ vdW for dimers and 0.25 × for trimers, so O–H and O–O at ice-X distances (~1.15–1.2 Å O–H, ~2.2–2.4 Å O–O) appear only as isolated fragments, never in a dense lattice. No ice VII/X, superionic water or high-pressure H–O phases are listed. H2O appears only as molecular crystals and fragments (SHIFTML) and as an adsorbate (MAD-cat). Possible exceptions are MC3D entries of ambient ice polymorphs, which I did not check. OMat24 pretraining adds PBE cells with cell deformations of μ 1%, σ 4–5% (a few-σ tail at most reaches V/V0 ≈ 0.7–0.8) and 1000/3000 K AIMD at Alexandria densities. Ice VII at 50 GPa is already at V/V0 ≈ 0.53, and 150 GPa is ≈ 0.40, far outside any sampled strain. So the teacher extrapolates throughout the ice VII/X regime. On top of that, PET-MAD-1 is PBEsol, a GGA. GGAs underestimate the proton double-well barrier (PBE is ~10 GPa low against SCAN) and over-delocalise H-bonded protons, so a static PBEsol teacher should symmetrise early relative to r2SCAN. The NQE-free classical symmetrisation pressure is the relevant comparison (~80–116 GPa for PBE). The one published uMLIP pressure benchmark (no PET-MAD) shows errors growing ~10× from 0 to 150 GPa and traces this to training data, not architecture.

Blocked sources: see BLOCKED.txt (Nature 1987/1998/1999 pages require login; HAL Klotz PDF behind a bot challenge).

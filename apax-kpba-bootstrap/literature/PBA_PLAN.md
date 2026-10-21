# K-PBA bootstrap plan: verification and improvements (2026-10-20)

Scope: K_xFe[Fe(CN)6] (x = 0..2), and K2Mn[Fe(CN)6] as an option. The teacher is PET-MAD v1.6 (pet-jax, CPU).

"Verified" means the number was read in the primary source: a CIF, SI table, full text or abstract. "Derived" means it was computed by us from verified numbers. "Secondary" means it was quoted by another paper and not read at the origin. "Unverified" means it was not checked. Detailed literature tables, each with its "how verified" column, are in `agentA/structures_findings.md` (structures) and `agentB/observables_findings.md` (observables). Structural parameters are in `structures.json`.

## 0. Decisions that change the v0 generator

1. **The teacher is non-spin-polarised.** It forces high-spin Fe–N and Mn–N into low-spin-like distances (§1, §2):
   - Fe–N: 1.91 Å vs 2.04 Å in PB (Fe(III) HS) and 2.17–2.23 Å in PW (Fe(II) HS).
   - Mn–N: 1.93 Å vs 2.22–2.26 Å.
   - Lattice parameter: 1.5–3% short.
   
   The teacher's surface is self-consistent, so it is fine for stress-testing the student. Compare student MD with the teacher, and with experiment only through trends and dimensionless ratios. **K2Mn[Fe(CN)6] is the worst case** (Mn–N 13% short) and should not be compared with experiment.
2. **Use PET-MAD-m, not -s, for anything involving C≡N.**
   - In the framework, -s puts the CN stretch about 180 cm⁻¹ too low: 1886–1996 cm⁻¹ harmonic at Γ. -m gives 2108–2169 cm⁻¹.
   - In molecules both are fine (HCN 2175 / 2155 cm⁻¹; experimental fundamental 2089).
   - So the -s error is specific to the framework. Do not use -s for screening labels.
3. **K is off-centre and the K-rich phase is P2₁/n.** In experiment, K is not at 8c: in P2₁/n K2Fe[Fe(CN)6] it is displaced 0.47 Å, mainly along a cubic ⟨110⟩ direction (the monoclinic b axis), and the cell has cooperative octahedral tilts (CAT23).
   - In the teacher (-m), cubic K2Fe with K at 8c is a saddle: K drifts 0.24 Å along ⟨111⟩ and imaginary Γ modes remain.
   - Started from the experimental structure, the -m teacher keeps the slide (0.39–0.41 Å) and the tilted P2₁/n cell. That cell is still 1.0 meV/atom above the [111]-displaced cubic cell.
   - **Add a P2₁/n slide/tilt block, and K-displacement scans along ⟨100⟩, ⟨110⟩ and ⟨111⟩.** Do not rely on 8c-centred cubic starts alone.
4. **The K hop barrier is 0.95–1.25 eV in the teacher (m)**, with the saddle at the 24d window, as in DFT. Bulk K diffusion is therefore not observable in nanosecond MD of a defect-free crystal at 300–600 K.
   - Use the NEB barrier, high-T (≥ 900 K) hop rates against teacher MD, and K rattling/off-centring as the diffusion-related tests.
   - The experimental D (1e-11 to 1e-9 cm²/s) reflects defects, vacancies and water. It is not a fair target for the ideal crystal.
5. **Rattles must be bond-aware.** At 900 K the teacher's classical C≡N bond-length rms is only 0.03 Å; Fe–C/Fe–N is 0.08 Å.
   - An isotropic per-atom rattle σ gives a bond rms of σ√2 per axis, so any σ > 0.02 Å on C/N samples C≡N stretches beyond 900 K.
   - At ±0.05 Å the C≡N energy is already +0.12 to +0.15 eV per bond.
   - Use rigid-unit plus normal-mode sampling (§4).

## 1. Teacher level of theory

| item | value | source | verified |
|---|---|---|---|
| PET-MAD-1.6 training data | MAD-1.6 = MAD-1.5 + catalytic surfaces; r2SCAN; fine-tuned from OMat-trained PET | lab-cosmo.github.io/upet/latest/models.html | yes (docs page) |
| MAD-1.5 DFT settings | FHI-aims (v250806), r2SCAN, "tight" basis, k-density 8 Å⁻¹, Gaussian smearing 0.05 eV, **non-spin-polarised** ("using non-spin-polarized calculations neglects important physical effects"); no Hubbard U mentioned | Malosso et al., arXiv:2603.02089 | yes (HTML full text) |
| MAD-1.0 (PET-MAD v1.0) | QE 7.2 + SIRIUS, PBEsol, SSSP 1.2 efficiency, MV smearing 0.01 Ry, "all calculations were intentionally conducted without spin polarization" | Mazitov et al., arXiv:2506.19674 | yes (HTML) |
| OMat24 (pre-training) | VASP PBE / PBE+U (MP-style U for oxides and fluorides with Co, Cr, Fe, Mn, Mo, Ni, V, W) | arXiv:2410.12771 | yes for PBE(+U); spin treatment **not verified** |
| Released heads | PBE heads discarded; r2SCAN head only | arXiv:2603.02089 (via search summary) | partly |

Implications:
- Low-spin Fe(II)–C (S = 0) is described consistently.
- HS Fe(III)–N (S = 5/2), HS Fe(II)–N (S = 2) and HS Mn(II)–N (S = 5/2) are forced into a closed-shell, low-spin-like state. M–N bonds come out about 0.13–0.33 Å short, and the M–N force constants are too stiff, which also stiffens the lattice.
- The redox-driven M–N length change with K content (PW 2.20 Å vs PB 2.04 Å) is lost. In the teacher, Fe–N only changes 1.91 → 1.89 Å from K2 to K1.
- Magnetism (PB T_C = 5.6 K) and spin crossover are absent.

## 2. Teacher checks (CPU, float64)

Setup: FrechetCellFilter + FIRE, fmax 0.02 eV/Å. Scripts: `teacher_checks.py`, `p21n_check.py`, `hcn_control.py`, `k_hop_neb.py`, `thermal_rms.py`. Raw results: `teacher_{s,m}.json`, `p21n_{s,m}.json`, `mol_{s,m}.json`, `khop_{s,m}.json`, `rms_s_*.json`. Relaxed cells: `relaxed_*.extxyz`.

Residual stress after the cell relax is 0.05–0.2 GPa. With B ≈ 30–40 GPa this means a is uncertain by about ±0.2%.

### 2a. Lattice and bonds vs experiment

| system | model | a (Å) [pseudo-cubic] | Fe–C | C–N | M–N | K off-centre (Å) | exp a (Å) | exp Fe–C / C–N / M–N | Δa |
|---|---|---|---|---|---|---|---|---|---|
| K2Fe[Fe(CN)6], Fm-3m start (64 at.) | s | 9.948 | 1.885 | 1.166 | 1.924 | 0.04 | 10.06 (P2₁/n pseudo-cubic, CAT23) | 1.87–1.94 / 1.11–1.13 / 2.17–2.23 | −1.1% |
| same | m | 9.907 | 1.878 | 1.167 | 1.908 | 0.24 (along ⟨111⟩; imaginary Γ modes remain) | 10.06 | same | −1.5% |
| K2Fe[Fe(CN)6], P2₁/n from CAT23 (32 at.) | s | 9.926 (a, b, c 9.933, 7.148, 6.887) | 1.88 | 1.165 | 1.92 | 0.17 (slide collapses) | a, b, c 10.063, 7.276, 6.948; β 89.80 | same | −1.3% |
| same | m | 9.873 (9.880, 7.136, 6.825; β 89.66) | 1.875–1.879 | 1.167 | 1.908–1.915 | 0.39–0.41 (slide kept) | same | same | −1.9% |
| KFe[Fe(CN)6], Fm-3m, K on one 8c sub-lattice (60 at.) | s | 9.901 | 1.881 | 1.161 | 1.909 | 0.03 | 10.178 (Fm-3m, K-PB, secondary); 10.166 (PB, Pm-3m, COD 4343748) | 1.92 / 1.12–1.13 / 2.04 (PB, COD) | −2.7% |
| same | m | 9.870 | 1.877 | 1.163 | 1.895 | 0.04 | same | same | −3.0% |
| Fe[Fe(CN)6] (Berlin green, 56 at.) | s | 9.839 | 1.883 | 1.156 | 1.881 | – | 10.132 (A-free FeFe(CN)6, Gao 2022, secondary via agentB) | – | −2.9% |
| same | m | 9.865 | 1.888 | 1.159 | 1.886 | – | 10.132 | – | −2.6% |
| K2Mn[Fe(CN)6], P2₁/n from CAT23 | s | 9.947 | 1.88 | 1.168 | Mn–N 1.92–1.93 | 0.19–0.24 | 10.0991, 7.3349, 6.948 (CAT23) | 1.90–1.94 / 1.09–1.15 / 2.22–2.26 | −1.8% |
| same | m | 9.922 (9.928, 7.200, 6.833) | 1.872–1.876 | 1.169 | Mn–N 1.931–1.939 | 0.41–0.44 | same | same | −1.8% |

How the experimental values were obtained:
- Experimental bonds: derived from CAT23 SI coordinates (agentA), and from the Buser CIF (COD 4343748; read by us).
- KFe[Fe(CN)6] a: "soluble" PB with no vacancies does not really exist (Bueno 2008: about 25% vacancies, abstract only).
- Teacher C–N (1.16–1.17 Å) is about 0.04 Å longer than XRD (1.11–1.13 Å). XRD C–N is known to read short, so this alone is not alarming.

Energy comparison of the two K2 forms:
- The m teacher puts the P2₁/n tilted K2Fe at E/atom −8.41199 eV vs −8.41294 eV for the [111]-displaced cubic cell, i.e. +0.95 meV/atom (+15 meV per f.u.) above it.
- The s teacher instead has P2₁/n 0.64 meV/atom below its cubic cell.
- The teacher therefore barely distinguishes the cubic and P2₁/n forms. This soft, multi-well landscape is a good stress test for the student.

### 2b. C≡N stretch

| quantity | s | m | experiment | source | verified |
|---|---|---|---|---|---|
| Γ-point CN band, K2Fe[Fe(CN)6] (24 modes, harmonic) | 1886–1949 | 2108–2169 | IR 2060–2070 (PW, Fe(II)–CN–Fe(II)); ~2067 | Kulesza 1996 Anal. Chem. 68, 2442; 10.22146/ijc.99310 | yes (agentB) |
| Γ-point CN band, KFe[Fe(CN)6] | 1937–1996 | not computed | IR 2070 (PB); Raman 2091 / 2152 (secondary), 2094.7 / 2157.3 (FeHCF film) | Kulesza 1996; arXiv:2609.26008 | yes |
| Γ-point CN band, Fe[Fe(CN)6] | 1950–1994 | not computed | IR 2163–2169 (BG, Fe(III)–CN–Fe(III)) | Kulesza 1996 | yes |
| local C–N pair stretch (±0.05 Å, quartic fit), K2 / K1 / K0 | 1885 / 1937 / 1948 | 2122 / 2137 / 2139 | – | this work | – |
| C–N pair force constant k, K2 / K1 / K0 | 84.5 / 89.2 / 90.2 eV/Å² | 107.1 / 108.6 / 108.8 eV/Å² | – | this work | – |
| ΔE at d ± 0.05 Å (stretch / compress), K2 | +0.096 / +0.130 eV | +0.121 / +0.150 eV | – | this work | – |
| HCN gas CN stretch (control) | 2175 | 2155 | 2089 (fundamental) | NIST CCCBDB HCN | yes (CCCBDB page) |

Comparing m with experiment:
- Harmonic and fundamental differ by about +25 cm⁻¹ for CN (anharmonicity; unverified textbook estimate). So m overestimates the PW CN frequency by about 20–80 cm⁻¹.
- m's oxidation-state shift (K2 → K0: +15 cm⁻¹ in the local mode) is much smaller than the experimental ~+100 cm⁻¹ (2065 → 2165). This is a consequence of having no spin and no charge localisation.

### 2c. K hop through the cage window (NEB, fixed cell, CI-NEB, 7 images, fmax 0.05)

| path | s barrier (eV) | m barrier (eV) | saddle | literature | source | verified |
|---|---|---|---|---|---|---|
| KFe[Fe(CN)6] (K4 cell), 8c → empty 8c (ΔE_end = +0.30 eV: different K ordering) | 1.46 | 1.25 (ΔE_end +0.21) | (¼, ¼, 0) or (¼, ¼, ½) = 24d window, both models | 0.874 eV single hop; 0.978 eV/ion concerted; PBE+U+D3, KFe[Fe(CN)6] | 10.1021/jacs.5c05274 | yes |
| K2Fe[Fe(CN)6] with one K vacancy (K7 cell), symmetric hop | 1.02 | 0.95 | 24d | 1.13 eV (PW cage), 1.20 / 0.80 eV (pristine / anion-defective) | cited in 10.1021/jacs.5c05274 | secondary |
| K2Mn[Fe(CN)6] | – | – | – | 0.53 eV (defect-free), 0.65 eV (defective) | 10.1038/s41467-021-22499-0 | yes |
| EIS activation energy, K-PBA film | – | – | – | 0.647 eV | cited in 10.1021/jacs.5c05274 | secondary |
| DFT-MD, pristine KFe[Fe(CN)6] | – | – | – | no K diffusion at 300–700 K (MSD < 1.2 Å²); hops only along an [Fe(CN)6]-vacancy channel at 700–800 K | 10.1021/jacs.5c05274 | yes |

### 2d. Teacher thermal amplitudes (s, classical harmonic, Γ Hessian of the 64- and 60-atom cells)

| T (K) | C≡N bond rms (Å) | Fe–C bond rms (Å) | Fe–N bond rms (Å) | K rms (Å) |
|---|---|---|---|---|
| 300 | 0.017 | 0.045–0.047 | 0.048–0.051 | 0.48–0.55 |
| 600 | 0.024–0.025 | 0.063–0.067 | 0.067–0.072 | 0.67–0.78 |
| 900 | 0.029–0.030 | 0.077–0.082 | 0.082–0.088 | 0.82–0.95 |

Per-atom rms of C/N/Fe (0.15–0.5 Å) is dominated by soft rigid-unit and K modes (lowest Γ modes 23–50 cm⁻¹). It is a harmonic over-estimate and is not useful as a rattle σ. The values are from s. m is stiffer for C≡N (k is 25% higher), so its C≡N rms would be about 11% smaller.

## 3. Experimental structures (details and coordinates in `structures.json`)

| composition | space group | a, b, c (Å), β | K site | vacancies / water | source | verified |
|---|---|---|---|---|---|---|
| Fe4[Fe(CN)6]3·14H2O (PB) | Pm-3m (vacancy-ordered Fm-3m) | 10.166 | – | 25% [Fe(CN)6] vacancies; coordinated H2O at 24e-type sites next to Fe(III) (Fe–O 2.135 Å); zeolitic O at (0.2608, 0.2608, 0.2608) | Buser 1977 Inorg. Chem. 16, 2704; COD 4343748 | yes (CIF) |
| K2Fe[Fe(CN)6] (refined K1.946) | P2₁/n | 10.063, 7.276, 6.948; 89.80°; V 508.8 Å³, Z = 2 | (0.7499, 0.5615, −0.0197); 0.47 Å from cage centre (0.75, 0.5, 0) | water not quantified; vacancy not measurable by ICP | Cattermull et al. JACS 145, 24249 (2023), 10.1021/jacs.3c08751, SI Table S6 | yes (SI) |
| K2Fe[Fe(CN)6] at high T | P2₁/n → P4/mnc at 599 K; decomposes at 778 K | – | slides switch off | – | same, SI Table S14 | yes |
| K2Mn[Fe(CN)6] (K1.974) | P2₁/n | 10.0991, 7.3349, 6.948; 90.01° | (0.7517, 0.5652, −0.0204); 0.50 Å off-centre | – | same, SI Table S7 | yes |
| K2Mn[Fe(CN)6] at high T | P2₁/n → P4/mnc at 797 K | at 795 K: 10.320, 7.320, 7.206 | – | – | same, SI Tables S9, S10, S14 | yes |
| K1.94Mn[Fe(CN)6]0.994·0.08H2O | P2₁/n | 10.0912, 7.3243, 6.9442; 90.016° | printed coordinates are inconsistent (Fe–C 2.1–2.6 Å) → do not use | 0.6% vacancy, 0.08 H2O | Deng 2021 Nat. Commun. 12, 2167, 10.1038/s41467-021-22499-0 | yes |
| KMn literature range | – | – | – | vacancies 0.006–0.13; H2O 0.08–1.43 per f.u. | Deng 2021 SI Table 4 | yes |
| K1.90Mn[Fe(CN)6]0.985·1.1H2O (neutron) | Pc supercell | – | 4 K sites | water in K cavities, occupancy 0.13–0.46 | Morozova 2021 Crystals 11, 895 | yes (SI CIFs, agentA) |
| K-PB (Fe(III)4[Fe(II)(CN)6]3 + KOH) | Fm-3m | 10.178 | K at 24e near coordinated water | ~25% vacancies | Bueno 2008 / Mansour 2021 | secondary |
| K2/3Cu[Fe(CN)6]0.68·2.2H2O (analogue) | Fm-3m | 10.0487 (K-free 10.1314) | 32f (x 0.224) and 48g, not 8c | C 24e x 0.1917; N 24e x 0.3061 | Ojwang (OSTI 1302943) | yes |
| K1.61Fe[Fe(CN)6]0.88·0.43H2O | P2₁/n | SI not obtained | – | 12% vacancies; 1.1% lattice and 3.5% volume change on cycling | Shu 2023 Small, 10.1002/smll.202207080 | yes (main text) |
| a(x) for K_xFe[Fe(CN)6] | **not found** as a table | – | – | – | – | – |

Check of the v0 generator: Fe(C) at 4a, Fe(N) at 4b, C/N at 24e and a ≈ 10.2 Å match the Buser and Ojwang data for the cubic, K-poor phases. Recommended changes:
- Use C x ≈ 0.189–0.192 and N x ≈ 0.30–0.31.
- For K ≥ 6 per cell, start from the CAT23 P2₁/n cell, not Fm-3m.
- Use the teacher's own a (9.87–9.95 Å) as the strain centre, not 10.2 Å. 10.2 Å sits +3% above the teacher minimum, under tension of about 1 GPa.

## 4. Proposed bootstrap blocks

Costs:
- All cells are 64 atoms (2 × P2₁/n 32-atom cells for the slide phase), or 128 atoms where noted.
- The m teacher costs about 1 s per 64-atom frame in float32 on CPU.
- About 4000 frames ≈ 1.2 h of teacher time on 16 threads, or about 4–5 h on 4 cores.

| block | frames | content | amplitudes / ranges | why |
|---|---|---|---|---|
| A. Compositions | across all blocks | K per 64-atom cell: 0, 1, 2, 3, 4, 5, 6, 7, 8. For each count, 3 K arrangements: max-dispersed, clustered, layered | – | odd counts give single K-vacancy and interstitial environments for hops; the K ordering energy at K4 was 0.30 eV between arrangements (NEB end-point difference) |
| B. Near-equilibrium normal-mode sampling | 900 (100 per K count) | sample from the teacher Γ Hessian (or from short teacher-free rigid-unit moves) at T = 150, 300, 600, 900, 1200 K | classical amplitudes as in §2d | covers the real thermal manifold without over-stretching C≡N; this is what student MD will visit |
| C. Rigid-unit + small rattle | 600 | rotate each Fe(CN)6 octahedron by 0–8° and translate it by 0–0.1 Å; then per-atom σ: C, N 0.010–0.025 Å; Fe 0.02–0.04 Å; K 0.05–0.3 Å | C≡N rms ≤ 0.035 Å (≈ 1000 K) | soft tilt and rattling modes need large amplitudes, stiff bonds need small ones; an isotropic σ cannot do both |
| D. K off-centring scans | 400 | K displaced 0–0.8 Å along ⟨100⟩ (toward the 24d window), ⟨110⟩ (the experimental slide direction) and ⟨111⟩ (the teacher's cubic drift / 32f); coherent and random patterns; K 4, 6, 8 | 0.1 Å steps plus a small rattle | experiment shows a 0.47–0.50 Å slide; the teacher has imaginary K modes at 8c; the multi-well shape is the key physics |
| E. P2₁/n slide/tilt phase (K2Fe, optional K2Mn) | 400 | linear interpolation 0 → 1.2 × the CAT23 slide/tilt distortion (mode amplitude), ferro and anti-ferro slide patterns, plus the P4/mnc tetragonal variant; strained ±3% | as stated | the experimental ground state; transitions at 599 K (K2Fe) and 797 K (K2Mn) are MD targets |
| F. Volume / strain | 500 | isotropic a from 0.94 to 1.08 × the teacher a (9.3–10.7 Å; includes exp 10.06–10.18); deviatoric ε from −3% to +3% (6 Voigt); combined with block-B rattles | 300 K | B is about 30–40 GPa (exp PBA analogues) and the P range of interest is ±2 GPa (PBA transitions at 0.2–3 GPa) |
| G. Stiff-bond probes | 300 | stretch or compress 1–3 individual C≡N bonds by −0.08 to +0.10 Å, and Fe–C / Fe–N by ±0.15 Å, in otherwise thermal frames | ΔE up to about 0.3 eV per bond | pins curvature and anharmonicity (cubic term −120 to −145 eV/Å³) that set the CN frequency; the student is judged on it |
| H. K hops | 300 | NEB images 8c → 24d → 8c, single and concerted, at K 1, 4, 7 per cell and at the vacancy-channel geometry; each image rattled at 300 K (block-C amplitudes) | 9 images per path | barrier about 1 eV (teacher), saddle at 24d; the student needs the saddle region for rare-event and high-T MD |
| I. [Fe(CN)6] vacancies + water | 400 | 1 or 2 Fe(C)(CN)6 vacancies per 64-atom cell (25 / 50% of 4a), with each exposed Fe(N) capped by H2O at 24e-type positions (Fe–O 2.14 Å); 0–1 zeolitic H2O per cage at 8c / 32f (PB O at (0.261, 0.261, 0.261)); water as rigid units with ±10° rotations | rattles as in block C | real K-PBAs have 1–25% vacancies and 0–1.4 H2O per f.u.; bare under-coordinated Fe is unphysical. Keep the K count charge-consistent: removing [Fe(CN)6]⁴⁻ removes 4 K or oxidises Fe |
| J. CN flips | 50 | 1 flipped C≡N per cell (Fe–N≡C–Fe linkage isomer), relaxed 0–5 steps plus rattle | – | a small repulsive / out-of-distribution regulariser; keep under 2% |
| K. High-T / out-of-distribution | 150 | block-C rattles ×2 at 1500 K equivalent; K pairs closer than 3 Å | – | stops the student from collapsing during the MD tests |

Total: about 4000 frames.

Drop or shrink from v0: the "volume scan" on Fm-3m alone (merge it into F), and the 10.2 Å strain centre.

Test MD protocol (student, then teacher MD spot checks):
- NVT and NPT at 150, 300, 600, 800, 1000 K and P = 0, 0.5, 1, 2 GPa.
- 64- and 512-atom cells.
- Time step 0.5 fs, because of C≡N at about 2100 cm⁻¹ (period 16 fs).

## 5. MD observables to judge the student

| observable | reference value | units / conditions | source | verified |
|---|---|---|---|---|
| CN stretch from the VACF of C/N | m teacher Γ: 2108–2169 (K2), local mode 2122 / 2137 / 2139 for K2 / K1 / K0; exp 2060–2070 (PW), 2070 (PB), 2163–2169 (BG) | cm⁻¹, 300 K | this work; Kulesza 1996 | teacher yes; exp yes |
| Fe–C stretch / Fe–C≡N bend region | Raman 510 (Fe–C), 276 (bend), PB; far-IR 450–650 | cm⁻¹ | Moretti & Gervais 10.1002/jrs.5366 | secondary |
| a(T), α | A-free FeFe(CN)6: −4.26e-6 K⁻¹ (NTE, 100–450 K); Na0.8: +0.40e-6; Na1.5: +4.00e-6 K⁻¹ (alkali makes PTE); no measured α for K-PBAs | 1e-6 K⁻¹ | 10.1039/c8dt05111a; 10.1016/j.rinp.2022.105410 | yes |
| P2₁/n → P4/mnc on heating | K2Fe: 599 K; K2Mn: 797 K | K | CAT23 | yes |
| K slide amplitude | 0.47 (K2Fe), 0.50 (K2Mn) at RT; m teacher 0.39–0.44 | Å | CAT23; this work | yes |
| Pseudo-cubic a | teacher m: 9.87–9.92 (K2), 9.87 (K1, K0); exp 10.06 (PW), 10.17–10.18 (PB), 10.13 (A-free) | Å, 0 K / RT | this work; CAT23; COD 4343748 | yes / secondary |
| Bulk modulus | PBA experiments 31–42 GPa (Mn/Cs/Rb Pt/Co cyanides); defect-rich 6.5–14 GPa; no K-PBA value | GPa | 10.1021/jacs.0c13181; 10.1039/d3sc06912e | yes |
| K hop barrier (student NEB vs teacher) | teacher s: 1.02 (vacancy hop), 1.46 (K4); m: 0.95 (vacancy hop), 1.25 (K4); DFT 0.87 (KFe), 0.53 (K2Mn) | eV | this work; 10.1021/jacs.5c05274; 10.1038/s41467-021-22499-0 | yes |
| K MSD / D | DFT-MD: no K diffusion in pristine KFe[Fe(CN)6] at ≤ 700 K; exp GITT 1e-11 to 1e-9 cm²/s (defective, RT) | – | 10.1021/jacs.5c05274; 10.1002/advs.202305467 | yes |
| Stability | no C≡N breaking or K escape in 1 ns NPT at 1000 K | – | – | – |

## 6. Open gaps

- No measured α, B0 or a(x) table for K_xFe[Fe(CN)6].
- No primary Berlin green cell.
- The OMat24 spin treatment was not verified.
- The Buser DOI (10.1021/ic50177a008) was not checked.
- CH3CN control frequencies (2237 / 2290 cm⁻¹ for s / m) have no verified experimental reference here.

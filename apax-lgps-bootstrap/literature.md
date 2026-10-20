# Li10GeP2S12 (LGPS): reference data for MLIP-MD comparison

Compiled 2026-10-19. **EXP** = experiment, **SIM** = simulation.
"Verified" means I read the number in the primary source myself. The access route is given each time: OA = open-access full text, AAM = author manuscript, SI = supplementary information, ABS = publisher or OpenAlex abstract only. "Secondary" means another paper quotes the number and I could not open the original.
Downloaded files are in `~/scratch/untrusted/lgps_lit/<paper>/`.

Units: 1 eV = 96.49 kJ/mol, so 24 kJ/mol = 0.249 eV.

---

## 1. Kamaya et al., Nat. Mater. 10, 682 (2011), doi:10.1038/nmat3066 (EXP)

| Quantity | Value | T | Method / sample | Where | Verified |
|---|---|---|---|---|---|
| σ (RT) | 12 mS cm⁻¹ | 27 °C | AC impedance; **total (bulk + grain boundary)** on a pellet | Abstract; Fig. 1 caption ("sum of the grain boundary and bulk conductivities"); Fig. 3 caption ("12 mS cm⁻¹ at 27 °C") | yes (abstract and figure captions on nature.com) |
| Lattice (NPD Rietveld) | a = 8.69407(18) Å, c = 12.5994(4) Å, V = 952.349(6) Å³; P4₂/nmc | 27 °C | Neutron powder diffraction (KENS) | SI Table 1 | yes (SI PDF) |
| Li occupancies (NPD; 3 Li sites, no Li4) | Li1 16h 0.691(5); Li2 4d 1.000(8); Li3 8f 0.643(5). Ge1/P1 on 4d 0.515/0.485; 2b is all P | 27 °C | NPD | SI Table 1 | yes (SI) |
| Ea | 0.24 eV | ? | total σ, Arrhenius | main text (paywalled) | **no**. Kuhn 2013 EES and Mo 2012 both quote 0.24 eV from this paper. "24 kJ/mol" (0.249 eV) is the same number in other units. |
| T range of the σ(T) measurement | not seen | — | — | main text, Fig. 1a | **no** (paywalled) |

Kwon 2015 (below) quotes the Kamaya reference value as 1.20 × 10⁻² S cm⁻¹.

## 2. Kuhn, Köhler, Lotsch, PCCP 15, 11620 (2013), doi:10.1039/c3cp51985f (EXP, single crystal)

Source: OA full text (LMU epub 22673), checked against COD CIFs 7215190 and 7215191.
Deposition numbers in the paper are CSD-425992 (298 K) and CSD-425993 (100 K).
COD 7215190 = 100 K and 7215191 = 298 K. Both CIFs list `_cell_measurement_temperature 273(2)`, which looks like a default value; `_diffrn_ambient_temperature` gives the real temperature.

| Quantity | 298 K | 100 K | Verified |
|---|---|---|---|
| a (Å) | 8.7187 (CIF 8.719(3)) | 8.6521 (CIF 8.652(3)) | yes |
| c (Å) | 12.6385 (CIF 12.639(4)) | 12.5816 (CIF 12.582(4)); the paper's Table 1 misprints this as "a =" | yes |
| V (Å³) | 960.72 | 941.84 | yes |
| ρ calc (g cm⁻³) | 2.035 | 2.076 | yes |
| Li1 16h occupancy | 0.466(18) | 0.474(16) | yes |
| Li2 4d occupancy | 0.86(6) | 0.89(5) | yes |
| Li3 8f occupancy | 0.74(5) | 0.72(4) | yes |
| Li4 4c (0,0,~0.251) occupancy (new site) | 0.81(7) | 0.77(6) | yes |
| Ge1/P1 4d | 0.5/0.5 (fixed) | 0.5/0.5 | yes |

These occupancies sum to about 20 Li per cell (Z = 2) at both temperatures (my arithmetic).
Mean linear expansion from these two cells, 100 to 298 K (my arithmetic): da/a ≈ 3.9 × 10⁻⁵ K⁻¹ and dc/c ≈ 2.3 × 10⁻⁵ K⁻¹.

Conclusions stated in the paper:
- Li4 sits between the c-channels and links two Li1 sites in neighbouring channels; the Li1–Li4 jump is 2.9 Å.
- Li4's ellipsoid lies perpendicular to c, so the authors argue Li4 is diffusion-active.
- Li1–Li1 distance is 1.6 Å, so Li1 occupancy should be ≤ 0.5.
- Their conclusion: diffusion is "less anisotropic than previously claimed".
- Synthesis: crystals from a 1023 K anneal; the product was tetragonal plus orthorhombic LGPS.

### 2b. Kuhn, Duppel, Lotsch, Energy Environ. Sci. 6, 3548 (2013), doi:10.1039/c3ee41728j (EXP, PFG-NMR, relaxometry, impedance)

Source: OA full text (LMU epub 22657). The SI (with the PFG data and details) could not be downloaded (404).

| Quantity | Value | T | Method / sample | Verified |
|---|---|---|---|---|
| Ea of Li self-diffusion D_tr | **0.21(1) eV** for LGPS (0.23(1) eV for Li7GePS8) | range not stated in the text; PFG data are only in Fig. 2B | ⁷Li PFG-NMR, Stejskal–Tanner 3D fit; polycrystalline powder (420 °C synthesis, small orthorhombic impurity) | yes |
| Numerical D_tr values | **only in Fig. 2B**; no numbers in the text | — | — | no numbers; digitise the figure if needed |
| Ea from NMR jump rates | 0.22(1) eV; attempt frequency τ⁻¹ ≈ 2 × 10¹² s⁻¹ | — | ⁷Li R1 (4.7 and 9.4 T) and motional narrowing; the R1 maxima are from Li7GePS8 | yes |
| Motional-narrowing onset | T_onset ≈ 135 K; τ⁻¹ ≈ 1.4 × 10⁴ s⁻¹; Waugh–Fedin estimate Ea ≈ 0.22 eV | 135 K | static ⁷Li NMR | yes |
| Quadrupolar averaging | τ⁻¹ ≈ 1.1 × 10⁵ s⁻¹ | 150 K | static ⁷Li NMR | yes |
| σ (RT), total | 9 mS cm⁻¹ for LGPS (7 for Li7GePS8) | RT | impedance, pellet | yes |
| Ea, bulk (low-T bulk combined with high-T total) | 0.22(2) eV | 110–450 K overall | impedance | yes |
| Ea, grain boundary | 0.30(2) eV; matters below 250 K | <250 K | impedance | yes |
| Slope of total σ | ~0.2 eV above RT, ~0.3 eV at low T | — | impedance | yes |
| Lattice (XRPD Rietveld) | a = 8.714(2) Å, c = 12.607(3) Å; Ge1/P1 = 0.54(2)/0.46(2) | RT | XRPD | yes |
| Haven ratio H_R and correlation factor f | "both in the order of 1" (D_tr = f·D_uc = H_R·D_σ, using a = 2 Å) | — | PFG vs relaxometry vs σ | yes, qualitative only |

PFG D_tr agrees with Mo et al.'s D_ab extrapolation. The authors saw no sign of the ultrafast D_c.

## 3. Mo, Ong, Ceder, Chem. Mater. 24, 15 (2012), doi:10.1021/cm203303y (SIM, AIMD)

Source: AAM (MIT DSpace 1721.1/80290, fetched through the Wayback Machine because DSpace returned 429). The SI was not available.
Parameters marked † come from the companion paper Ong et al., EES 6, 148 (2013), doi:10.1039/c2ee23355j (AAM, MIT DSpace 1721.1/81283), which uses the same protocol for the same group's LGPS runs. The 2012 AAM defers these details to its SI.

| Quantity | Value | Verified |
|---|---|---|
| Functional | PBE-GGA, PAW, VASP (HSE06 used only for the DOS) | yes |
| Structure | lowest-DFT-energy ordering among the 10 lowest-Ewald orderings | yes |
| T range | 600–1500 K | yes |
| Time step / run length | 2 fs; 40 ps diffusion runs ("convergence achieved with approximately 40 ps") | yes |
| Cell | 1 unit cell (50 atoms), Γ-only, 280 eV cutoff for sulfides; 2×2×1 tested and judged converged; NVT Nosé–Hoover; 2 ps heat-up plus 10 ps equilibration; 40–400 ps production | † yes (Ong 2013) |
| Ea, overall D | **0.21 eV** | yes |
| Ea along c / in ab | 0.17 eV / 0.28 eV | yes |
| σ (300 K), extrapolated | **9 mS cm⁻¹ (range 2–40)**; ~40 mS cm⁻¹ along c, 0.9 mS cm⁻¹ in ab; D_c ≈ 100 × D_ab at 300 K | yes |
| Numerical D(T) | only in Fig. 4; no numbers in the text | numbers not available |
| Ong 2013 Table 4 (same method) | Ea = 0.21 ± 0.04 eV; σ(300 K) = 13 mS cm⁻¹ | † yes |
| Ong 2013 PBE-relaxed cell | a = 8.561, b = 8.847, c = 12.929 Å, V = 979 Å³ (ordering breaks tetragonal symmetry) | † yes |
| Thermodynamics | metastable: 0 K decomposition to 2 Li3PS4 + Li4GeS4 is −25 meV/atom | yes |

## 4. Tracer / self-diffusion and Haven ratio

| Quantity | Value | T | Method | Source | EXP/SIM | Verified |
|---|---|---|---|---|---|---|
| Ea of D_tr | 0.21(1) eV | see §2b | ⁷Li PFG-NMR | Kuhn 2013 EES | EXP | yes |
| D_tr (LGPS) | 4.1 ± 2 × 10⁻¹² m² s⁻¹ | 300 K | NMR, as quoted by Huang et al. | Huang et al., arXiv:2006.03320 (cites Kuhn PCCP 2013, ref 17, which has no NMR, so probably meant the EES paper or a digitised figure) | EXP (secondary) | **secondary only** |
| Ea of D_tr (LGPS, re-quoted) | 0.21(1) eV (Li10SnP2S12 0.25(1), Li11Si2PS12 0.20(1)) | — | PFG-NMR | Kuhn et al., arXiv:1402.4586 (PCCP 2014) | EXP | yes (arXiv) |
| Haven ratio, LGPS | "order of 1" | RT region | PFG vs σ | Kuhn 2013 EES | EXP | yes, qualitative |
| Haven ratio, Li10SnP2S12 and Li11Si2PS12 | "on the order of unity" | — | PFG vs σ | arXiv:1402.4586 | EXP | yes |
| Haven ratio, LiTi2(PS4)3 (not LGPS-type) | ~0.24 (exp); 0.3 (AIMD). The paper says this is "on par with … LGPS" | 27 °C | PFG + impedance; AIMD | Di Stefano et al., arXiv:1708.02997 | EXP + SIM | yes (arXiv) |
| Haven ratio, LGPS (AIMD) | non-trivial H_R, with Nernst–Einstein failing; numbers not seen. Another paper quotes it as Ea ≈ 0.18 eV | — | PBE AIMD, tracer vs charge D | Marcolongo & Marzari, PRMaterials 1, 025402 (2017), doi:10.1103/PhysRevMaterials.1.025402 | SIM | **abstract only**; no OA copy found |
| ⁷Li NMR Ea regimes | 0.09(2) eV (low T, intrachannel); 0.176(5) eV (>120 K, Li1–Li3); 0.279(3) eV (~163–250 K, 3D incl. Li4); 0.194(5) eV (>293 K); bulk σ Ea 0.31 eV | 90–500 K | ⁷Li SAE, T1, T1ρ at 7 T; polycrystalline | Marko … Wilkening, JACS 147, 38215 (2025), doi:10.1021/jacs.5c10283 (PMC12550845) | EXP | yes (PMC full text read through a summarising fetch; D values are only in its SI) |
| T1 peak | τ ≈ 1.4 ns (τ⁻¹ ≈ 7.3 × 10⁸ s⁻¹) | 370 K | ⁷Li T1 | same | EXP | same |
| QENS | no LGPS QENS paper with numbers found. A Yonemura et al. ECS/IMLB 2016 abstract mentions QENS without values | — | — | ecs.confex.com/ecs/imlb2016/webprogram/Paper78899.html | EXP | **not found** |

### Single-crystal and off-stoichiometric conductivity (EXP; useful as bulk-only references)

| Quantity | Value | T | Sample | Source | Verified |
|---|---|---|---|---|---|
| σ along [001] / [110] | 27 / 7 mS cm⁻¹ | RT | self-flux single crystal; impedance | Iwasaki et al., Chem. Mater. 31, 3694 (2019), doi:10.1021/acs.chemmater.9b00420 | yes (ABS) |
| Ea in both directions | ≈ 0.3 eV (0.29 eV as quoted by Yajima 2021) | RT range | same | same | yes (ABS) |
| Single-crystal neutron diffraction (RT and 10 K) | composition refined to Li10.34Ge1.34P1.66S12 (δ = 0.34); Li3 splits at 10 K; OPP single-ion barrier along c 0.09 eV, Li1–Li4–Li1 along [110] 0.35 eV | RT, 10 K | ⁷Li-enriched single crystal, SENJU/J-PARC | Yajima et al., JMCA 9, 11278 (2021), doi:10.1039/d1ta00552a (CCDC 2051650/1/3) | yes (OA html through the Wayback Machine). The lattice numbers are in the ESI/CIF, which I did not fetch |
| Li10+δGe1+δP2−δS12: σ(300 K) for δ = 0.35 | 14.2 mS cm⁻¹ (bulk + GB 6.09 mS cm⁻¹ quoted separately) | 300 K | sintered pellet (873 K), Au electrodes, 100 Hz–15 MHz | Kwon et al., JMCA 3, 438 (2015), doi:10.1039/c4ta05231e | yes (OA html) |
| Ea, δ = 0.05 / 0.35 | 27 / 26 kJ mol⁻¹ (193–373 K); 12 / 7 kJ mol⁻¹ (573–673 K) | as stated | same | same | yes |
| NPD lattice, δ = 0.35 | a = 8.713023(5) Å, c = 12.639695(11) Å; Li1 0.473(11), Li2 1, Li3 0.75(2), Li4 4c 0.77(2) | 300 K | NPD Rietveld | Kwon 2015 Table 2 | yes |

## 5. Lattice parameters vs temperature (thermal expansion)

| Data | Values | Source | EXP/SIM | Verified |
|---|---|---|---|---|
| Two single-crystal cells | 100 K and 298 K (§2) | Kuhn 2013 PCCP | EXP | yes |
| NPD and synchrotron lattice vs T for Li10+δGe1+δP2−δS12 (δ = 0.05, 0.35) | Fig. 7 only; lattice and volume rise with T up to 750 K with no phase change. T ranges: 4.8–300 K and 10–750 K (the text and caption assign these ranges to opposite compositions). Li1 and Li4 occupancy rise and Li3 falls with T; Li2 stays at 1.0 | Kwon 2015 | EXP | yes (qualitative; **numbers are only in the figure**) |
| Temperature-dependent XRD; anisotropic expansion; Debye temperature from speed of sound; partial decomposition to Li4P2S6 on heating | numbers not seen | Weber et al., Chem. Mater. 28, 5905 (2016), doi:10.1021/acs.chemmater.6b02424 | EXP | **abstract only** (paywalled) |
| Linear α (300 K, from Weber 2016) | 3.5 × 10⁻⁵ K⁻¹ in one place and 3.4 × 10⁻⁵ K⁻¹ in another, both in the same paper; linear below ~700 K, slightly anisotropic above | Huang et al., arXiv:2006.03320, quoting Weber 2016 | EXP (secondary) | **secondary only** |
| DP-PBE / DP-PBEsol α (NpT, 1 ns) | 3.2 × 10⁻⁵ K⁻¹ | Huang et al. arXiv:2006.03320 | SIM | yes |

## 6. Decomposition / melting, and MLIP studies

- **Decomposition (EXP):** Weber 2016 abstract says LGPS partially decomposes to Li4P2S6 at elevated temperature, which explains the loss of conductivity on heating. The onset temperature is not in the abstract and was not verified.
- **Melting point:** no source found.
- **Huang, Zhang, Wang, Zhao, Cheng, E, arXiv:2006.03320 (DP-GEN; SIM):**
  - Setup: PBE and PBEsol Deep Potential models; 900-atom cells (Li360Ge36P72S192); 2 fs time step; NVT at experimental lattice parameters; 1 ns runs above 400 K and 10 ns at 300 K.
  - Finite size: a 2×2×1 cell (200 atoms) overestimates D by 10–100× at 300 K.
  - Cation disorder raises D by 2–4× at low T.
  - D(300 K) for LGPS is 19 ± 2 × 10⁻¹² m² s⁻¹, against an experimental value of 4.1 × 10⁻¹² as they quote it.
  - Table II lattice parameters (a, b, c in Å): PBE 8.591, 8.879, 12.977; PBEsol 8.503, 8.811, 12.760; SCAN 8.534, 8.819, 12.917.
  - Verified (arXiv v. 28 Jan 2021; published version not checked).
- **Qi et al., Materials 17, 1810 (2024), doi:10.3390/ma17081810 (PMC11051406; SIM):**
  - Setup: linear-ridge Gaussian-basis ensemble MLIP; PBE; 50-atom cell; 500–900 K; 0.5–2 ns runs.
  - Ground-state ordering Zig_2: Ea 0.17 ± 0.008 eV (c 0.15, ab 0.25); σ(300 K) 43 mS cm⁻¹.
  - Ea across orderings: 0.09–0.22 eV.
  - Lowest-energy ordering puts 2 Li on Li4 sites.
  - Verified (PMC, read through a summarising fetch).
- **M3GNet vs AIMD (UMD student poster; SIM):** Ea 0.14 eV (M3GNet) vs 0.23 eV (AIMD). The source is a poster, not peer reviewed, and I saw only the search snippet.
- **r2SCAN-level LGPS studies:** none found.

---

## Not verified / gaps
1. Kamaya 2011: the main text (Ea, measurement T range, Arrhenius data) is paywalled. 0.24 eV is secondary (quoted by Kuhn 2013 EES and Mo 2012).
2. Numerical PFG D_tr(T) for LGPS: only in Kuhn 2013 EES Fig. 2B; the SI was unavailable. The 300 K value of 4.1 × 10⁻¹² m² s⁻¹ is secondary (Huang et al.).
3. Mo 2012 D(T) values and cell size: these are in the SI, which I could not get. The cell (1 unit cell, 50 atoms) is inferred from the companion Ong 2013 protocol.
4. Haven ratio for LGPS: the experiment gives only "order of 1" (Kuhn). Marcolongo & Marzari's AIMD numbers are paywalled.
5. Weber 2016 thermal expansion and decomposition temperature: abstract only. The α value is secondary.
6. Kwon 2015 lattice vs T: figure only.
7. No QENS numbers or melting point found for LGPS.
8. Yajima 2021 RT/10 K single-crystal lattice parameters: in the ESI/CIF (CCDC 2051650/1/3), not fetched.

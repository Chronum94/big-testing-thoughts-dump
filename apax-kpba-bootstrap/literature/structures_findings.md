# K-PBA structure findings (agentA, 2026-10-20)

"Verified" means I read the number in the source myself (full text, SI, CIF or xlsx). "Derived" means I computed it from verified numbers; the script is named. "Unverified" means it came from a secondary source or a snippet. Downloads are in `/home/chronum/scratch/untrusted/pba_lit/struct_A*/`.

Key sources:
- **[CAT23]** Cattermull, Roth, Cassidy, Pasta, Goodwin, "K-Ion Slides in Prussian Blue Analogues", JACS 145, 24249 (2023), doi 10.1021/jacs.3c08751. I read the main text on PMC, the SI pdf and the SI xlsx (fetched via the Europe PMC supplementaryFiles API).
- **[DENG21]** Deng et al., Nat. Commun. 12, 2167 (2021), doi 10.1038/s41467-021-22499-0. I read the main text on PMC and the SI pdf.
- **[MOR21]** Morozova et al., Crystals 11, 895 (2021), doi 10.3390/cryst11080895. I read the full text from the epub and the SI CIFs (neutron TOF).
- **[OJW]** Ojwang et al., "Preparation, structure characterization and properties of K-containing copper hexacyanoferrate" (Svensson group, accepted manuscript, OSTI 1302943; the journal DOI is not verified). I read the full text.
- **[MAN21]** Mansour et al., ECS J. Solid State Sci. Technol. 10, 103002 (2021) (OSTI 1842289). I read the full text.
- **[SHU23]** Shu et al., Small 2023, 2207080, doi 10.1002/smll.202207080. I read the main text; the SI was not available.
- **[LEP23]** Le Pham et al., J. Mater. Chem. A 11, 3091 (2023), doi 10.1039/d2ta08439b. I read the full text.
- **[BUSER]** COD 4343748 (Buser 1977). I read the CIF myself.
- **[BUENO08]** Bueno et al., J. Phys. Chem. C 112, 13264 (2008), doi 10.1021/jp802070f. I could only read the abstract summary from the UNESP repository; the full text is restricted.

## 1. K2Fe[Fe(CN)6] (Prussian white, K-rich)

| Quantity | Value | Units | Conditions | Source | Verified | How |
|---|---|---|---|---|---|---|
| Space group | P2₁/n (monoclinic) | – | room temperature (RT), synchrotron XRPD (I11) | CAT23 | yes | SI Table S6 |
| Composition (refined) | K1.946(2)Fe[Fe(CN)6] | – | RT; refined K occupancy | CAT23 | yes | SI Table S6. ICP cannot give the vacancy fraction because both TM sites are Fe (SI text) |
| a, b, c | 10.063(19), 7.276(5), 6.948(5) | Å | RT | CAT23 | yes | SI Table S6 |
| β | 89.7977(7) | ° | RT | CAT23 | yes | SI Table S6 |
| V | 508.8(12) | Å³ | RT. Table states Z=4, but the Fe count gives 2 K2Fe2(CN)6 per cell | CAT23 | yes | SI Table S6 |
| a, b, c, β, V (xlsx) | 10.06616, 7.27737, 6.95001, 89.79837, 509.12 | Å, ° | 299.4 K, sequential VT refinement | CAT23 | yes | SI xlsx, first K2Fe row |
| Pseudo-cubic a | 10.06 | Å | (2V)^(1/3) | derived | – | from V |
| Monoclinic→tetragonal (P4/mnc) | Tc = 599.3 K; tetragonal fraction 4% at 598 K, 20% at 647 K; decomposition Td = 778.2 K | K | heating | CAT23 | yes | SI Table S14 and xlsx |
| Coordinate: Fe1 (C-bound, LS) | (0, 0, 0) | frac | RT | CAT23 | yes | Table S6 |
| Coordinate: Fe2 (N-bound, HS) | (0.5, 0, 0) | frac | RT | CAT23 | yes | Table S6 |
| Coordinate: K | (0.7499(3), 0.56149(11), −0.01970(19)); Uiso 0.0314 | frac | RT. Ideal cavity centre (parent 8c) is (0.75, 0.5, 0) | CAT23 | yes | Table S6 |
| Coordinates: C1, C2, C3 | (−0.0322, 0.8580, 0.7744), (−0.0318, 0.2110, 0.8542), (0.8123, −0.0004, 0.0619) | frac | RT | CAT23 | yes | Table S6 |
| Coordinates: N1, N2, N3 | (−0.0437, 0.7540, 0.6584), (0.9485, 0.3353, 0.7609), (0.7045, −0.0312, 0.0913) | frac | RT | CAT23 | yes | Table S6 |
| K off-centre displacement | 0.47 (vector 0, +0.45 along b, −0.14 along c) | Å | RT | derived | – | `bonds_p21n.py` |
| Fe(LS)–C | 1.868, 1.906, 1.936 | Å | RT | derived | – | `bonds_p21n.py` |
| Fe(HS)–N | 2.167, 2.195, 2.232 | Å | RT | derived | – | `bonds_p21n.py` |
| C–N | 1.112–1.131 | Å | RT | derived | – | `bonds_p21n.py` |
| Shortest K–N | 3.00 | Å | RT | derived | – | `bonds_p21n.py` |
| Sample preparation | precipitated from citrate solution, air-dried at 70 °C overnight; water not quantified | – | – | CAT23 | yes | main text |
| Second sample (KFeHCF-E) | K1.61Fe[Fe(CN)6]0.88·0.43H2O, P2₁/n (GSAS). The control sample is K1.51Fe[Fe(CN)6]0.82·0.59H2O | – | EDTA-chelated coprecipitation | SHU23 | yes | main text. Its cell parameters are only in the SI, which I did not obtain |
| Strain on cycling (KFeHCF-E) | lattice change 1.1%, cell volume change 3.5%; solid solution; lattice expands on K extraction | % | in situ XRD | SHU23 | yes | main text |
| K1.83Fe[Fe(CN)6]·nH2O | monoclinic P2₁/c, stable to 250 °C | – | – | JINR conference abstract | no | search snippet only |

## 2. KFe[Fe(CN)6] ("soluble PB") and Berlin green Fe[Fe(CN)6]

| Quantity | Value | Units | Conditions | Source | Verified | How |
|---|---|---|---|---|---|---|
| "Soluble" KFe[Fe(CN)6] model | rejected: about 25% of [Fe(CN)6] is absent. Proposed formula Fe4(III)[Fe(II)(CN)6]3·[K⁺h·OH⁻h·mH2O] | – | electrodeposited films, synchrotron XRPD | BUENO08 | partly | abstract via repository (summarised by the fetch tool) |
| K site in K-rich PB | 24e, near the coordinated-water site (not the cavity) | – | – | BUENO08, as cited by OJW | secondary | OJW text |
| K-PB cubic cell | Fm-3m, a = 10.178 Å; PDF 04-016-2997, K0.525Fe1.75(CN)4.5(OH)0.525(H2O)2.98 (= (KOH)0.7Fe(III)4/3Fe(II)(CN)6·4.0H2O) | Å | – | MAN21, citing BUENO08 | secondary | MAN21 text |
| Prussian white (fully reduced) | "all Fe 2+ and each cavity contains one K" (Keggin & Miles 1936) | – | – | OJW | secondary | text |
| Berlin green Fe[Fe(CN)6] cell | **not found** | – | – | – | – | no primary source reached |

## 3. K2Mn[Fe(CN)6] (KMF)

| Quantity | Value | Units | Conditions | Source | Verified | How |
|---|---|---|---|---|---|---|
| Space group, composition | P2₁/n, K1.9736(18)Mn[Fe(CN)6] | – | RT, synchrotron | CAT23 | yes | SI Table S7 |
| a, b, c, β, V | 10.0991(15), 7.3349(4), 6.948(4), 90.0118(5), 514.75(9) | Å, °, Å³ | RT | CAT23 | yes | Table S7 |
| Coordinates: Fe, Mn | Fe (0, 0, 0); Mn (0.5, 0, 0) | frac | RT | CAT23 | yes | Table S7 |
| Coordinate: K1 | (0.7517(2), 0.56517(6), −0.02040(10)) | frac | RT | CAT23 | yes | Table S7 |
| Coordinates: C1, C2, C3 | (−0.0344, 0.8550, 0.7713), (−0.0356, 0.2136, 0.8554), (0.8174, −0.0133, 0.0667) | frac | RT | CAT23 | yes | Table S7 |
| Coordinates: N1, N2, N3 | (−0.0517, 0.7642, 0.6491), (0.9429, 0.3412, 0.7640), (0.7070, −0.0294, 0.1032) | frac | RT | CAT23 | yes | Table S7 |
| K displacement | 0.50 Å. The paper uses "a typical value d = 0.5 Å" | Å | RT | derived; CAT23 | yes | `bonds_kmf_cattermull.py`; main text |
| Fe–C | 1.90–1.94 | Å | RT | derived | – | `bonds_kmf_cattermull.py` |
| Mn–N | 2.22–2.26 | Å | RT | derived | – | `bonds_kmf_cattermull.py` |
| C–N | 1.09–1.15 | Å | RT | derived | – | `bonds_kmf_cattermull.py` |
| High-temperature phases | P2₁/n at 795 K: a 10.320, b 7.320, c 7.206 Å, β 89.924°, V 545 Å³. T′ phase P4/mnc at 803 K: a 7.308, c 10.276 Å, V 548.9 Å³. Tc = 797 K | – | heating | CAT23 | yes | SI Tables S9, S10, S14 |
| KMF-EDTA | K1.94Mn[Fe(CN)6]0.994□0.006·0.08H2O; P2₁/n; a 10.0912(3), b 7.3243(2), c 6.9442(2) Å, β 90.016(6)°; Rwp 1.48% | – | – | DENG21 | yes | SI Tables 2–3 |
| KMF-EDTA coordinates (warning) | K 4e (0.75165, 0.93746, 0.47994), occupancy 0.97; Fe 2a (0,0,0); Mn 2a (0, ½, ½). In the stated cell these coordinates give Fe–C 2.1–2.6 Å and C–N 0.9 Å, for every axis permutation and with both P2₁/n and P2₁/c. **Do not use them**; use CAT23 instead | – | – | DENG21 | yes (as printed) | `perm_check.py` |
| KMF-C | K1.72Mn[Fe(CN)6]0.92·1.43H2O; a 10.14201, b 7.25722, c 7.03956 Å, β 90.52128° | – | – | DENG21 | yes | SI Tables 1–2 |
| On charge (DENG21) | monoclinic → cubic (mid-charge) → tetragonal (end of charge); reversible. No volume % given | – | ex situ XRD | DENG21 | yes | main text |
| Phase boundaries in K2−xMn[Fe(CN)6] | monoclinic→cubic at x≈0.45; cubic limit x≈1.15; tetragonal from x≈1.85. Cubic phase is *larger*, tetragonal smaller | – | operando | Cattermull et al. arXiv 2509.04587 (Sci. Adv.) | yes | arXiv HTML text |
| Neutron model | Pc, 2×b supercell. ad sample at 300 K: a 6.9351(6), b 14.6289(13), c 12.2217(12) Å, β 124.491(4)°, V 1021.97 Å³. vd sample at 300 K: 6.9455, 14.6313, 12.2396 Å, 124.531°, V 1024.67 Å³ | – | TOF NPD | MOR21 | yes | Table 1 and CIF |
| Neutron composition | K1.90Mn[(OH)6]0.015[Fe(CN)6]0.985·1.14H2O (ad), ·1.03H2O (vd) | – | – | MOR21 | yes | text |
| Neutron bond lengths | Fe–C 1.87–1.95; Mn–N 2.11–2.31; C–N 1.13–1.24 | Å | 300 K | MOR21 | yes | CIF geom_bond loop |
| Neutron water | 8 partially occupied H2O per cell, all in the K cavities (occupancies 0.13–0.46) | – | 300 K | MOR21 | yes | CIF |
| Mn/Fe-mixed KMF ("MF21") | K1.67Mn0.65Fe0.35[Fe(CN)6]0.92·0.45H2O; P2₁/n a 10.009, b 7.228, c 6.921 Å, β 90.38°. Charged state: Fm-3m, a = 10.3478(3) Å | – | – | LEP23 | yes | text |
| Volume change, MF21 | +10.7% per formula unit on charge (expansion) | % | – | derived | – | from LEP23 cells |

## 4. a versus x, vacancies and water

| Quantity | Value | Source | Verified | How |
|---|---|---|---|---|
| Cu analogue, a versus K | Cu[Fe(CN)6]0.70·1.9H2O: a = 10.1314(2) Å. K2/3Cu[Fe(CN)6]0.68·2.2H2O: a = 10.0487(2) Å (−0.8%) | OJW | yes | Tables 3a/3b |
| Cu analogue, sites | C 24e x=0.1917, N 24e x=0.3061, coordinated O1 24e x=0.212; zeolitic O 8c, 32f (x 0.224), 48g. K over 32f (x 0.224, occupancy 0.04) and 48g (0.325, ¼, ¼; occupancy 0.03); no K at 8c | OJW | yes | Table 3b |
| PB water sites (Herren 1980 neutron) | zeolitic water at 8c and 32f; coordinated water at 24e next to Fe(III) at vacancies; maximum 14 H2O per cell (8 in cavities + 6 coordinated) | OJW, citing Herren, Inorg. Chem. 19, 956 (1980) | secondary | OJW text |
| Vacancy/water ranges in KMF (literature table) | vacancy 0.006–0.13 per Mn; H2O 0.08–1.43 per f.u. (e.g. Xue JACS 2017: K1.89, 0.92, 0.75H2O; Bie 2017: K1.75, 0.93, 0.16H2O) | DENG21 | yes | SI Table 4 |
| Low-vacancy K2Mn[Fe] | "is anhydrous". Interstitial water is replaced by K | Cattermull review arXiv 2107.07448 | yes | text |
| Slide amplitude | scales with K content; largest in low-vacancy, high-K PBAs | arXiv 2107.07448; CAT23 | yes | text |
| KxFe[Fe(CN)6] a(x) table | **not found**. Only the endpoint values in §1–2 | – | – | – |

## 5. Bond lengths, PB (Buser, COD 4343748, Pm-3m, a = 10.166 Å)

| Bond | Value (Å) | Basis |
|---|---|---|
| Fe(III,HS)–N | 2.038 | derived from CIF: N1 x=0.2005, N3 x=0.2995 |
| Fe(II,LS)–C | 1.918, 1.923 | derived from CIF: C3 x=0.1887, C1 x=0.3108 |
| C≡N | 1.121, 1.126 | derived from CIF |
| Fe(III)–O (coordinated water) | 2.135 | derived from CIF |
| MAN21 summary | Fe(II)–C 1.92; Fe(III)–N 2.03; Fe(III)–O 2.14 (PB). Fe(II)–C 1.92 in K4Fe(CN)6·3H2O; Fe(III)–C 1.94 in K3Fe(CN)6; C–N 1.151/1.154 (XRD) | verified in MAN21 text |
| PB CIF occupancies | Fe(II) 3d 0.911 / 1b 0.267 (vacancy disorder); zeolitic O4 32f (0.2608)³ occupancy 1; O2 3c-type (0.21, ½, ½) occupancy 0.733 | read in CIF |

HS Fe(II)–N in Prussian white is 2.17–2.23 Å (derived, §1), against 2.04 Å for HS Fe(III)–N in PB. LS Fe–C stays at 1.87–1.94 Å.

## 6. Coordinate files

- **COD.** No K–Fe–CN Prussian-blue-type entry exists. I queried K/Fe/C/N with ≤5 elements, Fe/C/N, and K/Mn/Fe/C/N. The only PB-like hit is 4343748, plus 1521296/6000335 (KMn, no coordinates).
- **MOR21 SI CIFs.** `struct_A/s001/Prussian_{10K,300K}{NOvac,vac}.cif`, space group Pc, every site 2a, with full x,y,z and occupancies. Example from 300K NOvac: Fe_1 (0, 0.6270, 0); Fe_2 (0.019, 0.1282, 0.013), occupancy 0.971; K_1–K_4 at (0.828, 0.601, 0.275), (0.783, 0.084, 0.228), (0.276, 0.855, 0.279), (0.260, 0.338, 0.259), occupancies 1, 1, 1, 0.81.
- **CAT23 xlsx.** Full variable-temperature cells and K coordinates for M = Mn, Fe, Co, Ni, Cd, dumped to `agentA/slides_xlsx.txt`. K2Ni (297 K): a 10.0329, b 7.2045, c 6.9488 Å, β 89.716°. K2Co: 10.0460, 7.2514, 6.9336 Å, 89.781°. K2Cd: 10.1766, 7.4124, 6.9984 Å, 89.839°.

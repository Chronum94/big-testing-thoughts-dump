# Reference observables for K-PBA MD (agent B, 2026-10-20)

"Verified" means I read the number myself in the source text listed (abstract, open-access full text, or arXiv PDF). Values passed on from a secondary source are marked "secondary". Downloads are in /home/chronum/scratch/untrusted/pba_lit/obs_B/ and the extraction scripts are in /home/chronum/scratch/pba/lit/agentB/.

## 1. K+ transport

| quantity | value | units | conditions | source DOI | verified | how |
|---|---|---|---|---|---|---|
| D(K+), GITT | 1e-11 to 1e-9 | cm2/s | KFeHCF (K-rich K2FeFe(CN)6 type) on carbon cloth, 2nd cycle, 2.5 M KFSI/TEP, RT, 50 mA/g pulses of 300 s | 10.1002/advs.202305467 | yes | EuropePMC full text (PMC10837388) |
| D(K+), PITT | reaches 1e-13 after full K extraction | cm2/s | K2-δMn[Fe(CN)6]1-γ·nH2O, 3-electrode cell | 10.3390/cryst11080895 | yes (abstract only) | DOAJ API abstract; MDPI full text blocked by bot protection |
| D(K+), "two plateaus" | ~1e-8 (abstract gives no units) | presumably cm2/s | K1.43Fe[Fe(CN)6]0.94·0.42H2O (PB25), vacancy-poor | 10.1002/bte2.20220027 | yes (abstract); units unverified | DOAJ API abstract |
| NEB barrier, K+ single hop 8c→8c (linear BC-BC) | 0.874 | eV/ion | KFe[Fe(CN)6] model (K4Fe4[Fe(CN)6]4), anhydrous, PBE+U(5 eV)+D3 | 10.1021/jacs.5c05274 | yes | PMC12291462 full text |
| NEB barrier, K+ concerted | 3.910 eV/cell (0.978 eV/ion) | eV | same | 10.1021/jacs.5c05274 | yes | same |
| NEB, Na+ / Li+ single hop (comparison) | 0.414 / 0.761 | eV | same model | 10.1021/jacs.5c05274 | yes | same |
| AIMD D*(K+) in pristine KFe[Fe(CN)6] | not measurable; MSD < 1.2 Å2 at 300, 500 and 700 K | – | DFT-MD, pristine anhydrous | 10.1021/jacs.5c05274 | yes | same |
| AIMD K+ with an anion-vacancy channel | D* only along the channel (z) at 700 K; x and z at 800 K; no number given in text | – | 4×2×2 cages with 2 [Fe(CN)5] defects, 20 ps runs | 10.1021/jacs.5c05274 | yes | same |
| AIMD D*(Na+), D*(Li+) (comparison) | Na 7.98e-6 (300 K), 3.81e-5 (700 K); Li 5.83e-6 (300 K), 1.05e-4 (700 K); Ea(MD) Na 0.070, Li 0.140 eV | cm2/s | same model | 10.1021/jacs.5c05274 | yes | same |
| K+ barrier, single K in a Prussian-white cage (earlier work by the same group) | 1.131 | eV | DFT-NEB | cited in 10.1021/jacs.5c05274 | secondary | ref not resolved |
| K+ NEB, anion-defective vs pristine framework (earlier work) | 0.80 vs 1.20 | eV | DFT-NEB | cited in 10.1021/jacs.5c05274 | secondary | ref not resolved |
| K+ Ea from EIS | 0.647 | eV | PBA film, K/Mn/Fe = 1.81:1:0.96, with anion vacancies | cited in 10.1021/jacs.5c05274 | secondary | ref not resolved |
| K+ migration barrier (DFT) | 0.53 (defect-free) vs 0.65 (defective, with water) | eV | K2Mn[Fe(CN)6] (K1.94Mn[Fe(CN)6]0.994·0.08H2O) | 10.1038/s41467-021-22499-0 | yes | PMC8041879 full text |
| Na+, Li+ Ea from EIS (Mn-PBA film, comparison) | Na 0.22, Li 0.48 (a = 10.56 Å) | eV | MnFe(CN)6 films | cited in 10.1021/jacs.5c05274 | secondary | – |

## 2. Thermal expansion (linear α)

| quantity | value | units | conditions | source DOI | verified | how |
|---|---|---|---|---|---|---|
| α, FeFe(CN)6 (Berlin-green type, A-free) | −4.26 (NTE) | 1e-6/K | 100–450 K (abstract); 100–475 K (Gao 2022 text) | 10.1039/c8dt05111a | yes | CityU abstract page; restated in Gao 2022 full text |
| α, Na0.8FeFe(CN)6 | +0.40 (near zero) | 1e-6/K | 100–475 K, cubic | 10.1016/j.rinp.2022.105410 | yes | open-access PDF |
| α, Na1.5FeFe(CN)6 | +4.00 (PTE) | 1e-6/K | 100–475 K, cubic; a goes from 10.13182(4) to 10.33152(3) Å | 10.1016/j.rinp.2022.105410 | yes | same; **inserting alkali turns NTE into PTE** |
| α, Fe3[Fe(CN)6]2·14H2O | −9.9(12.9), not significant | 1e-6/K | 123–298 K | 10.1016/j.jssc.2011.08.030 (arXiv:1104.3554) | yes | Table 3 of the arXiv PDF |
| α, Mn3[Fe(CN)6]2 / Co / Ni / Cu / Zn | +47.8 / +7.9 / +5.9 / −19.9 / −39.6 | 1e-6/K | same | same | yes | same |
| α, M2[Fe(CN)6] (Mn / Co / Ni / Cu / Zn) | +20.2 / +19.5 / +19.9 / +20.1 / +43.1 (all PTE); the Fe2 compound was unstable | 1e-6/K | same | same | yes | same |
| α, M3[Co(CN)6]2 | −19.6 (Fe) to −39.7 (Co); all NTE | 1e-6/K | same | same | yes | same |
| α, Fe[Co(CN)6] | about −1.5 ("nearly zero") | 1e-6/K | 4.2–300 K | 10.1021/ja044959o | number secondary; "nearly ZTE" verified | PubMed abstract (qualitative); number from Adak intro |
| α, MPt(CN)6 | NiPt −1.02(11) to CdPt −10.02(11) | 1e-6/K | 100–400 K | 10.1021/ja060916r | yes | PubMed abstract |
| Water dependence | sign of α changes with hydration in Fe3[Co(CN)6]2·xH2O | – | APS abstract (Nakotte group) | none | no | search snippet only |
| α for KFe[Fe(CN)6], K2Fe[Fe(CN)6] or K2Mn[Fe(CN)6] | **not found** | – | – | – | – | gap |

## 3. Temperature-driven transitions

| quantity | value | units | conditions | source DOI | verified | how |
|---|---|---|---|---|---|---|
| K2Mn[Fe(CN)6] P21/n → tetragonal P4/mnc (K-ion slides switched off) | Tc ≈ 800 K, first order; transient T′ phase between 795 and 980 K; second step near 900 K | K | synchrotron PXRD, 300–1000 K | 10.1021/jacs.3c08751 | yes | PMC10636749 full text |
| Size of K-ion slide displacement | ≈ 0.5 | Å | K-rich PBAs | same | yes | same |
| Trend across K2M[Fe(CN)6] | Tc lowest for the smallest M2+ | – | – | same | yes | same |
| K2Cu[Fe(CN)6] | slides vanish by about 250 °C (≈520 K); exsolved K2Fe[Fe(CN)6] is nearly cubic by about 425 °C, where its monoclinic distortion effectively vanishes | °C | VT-PXRD 30–450 °C | 10.1021/acs.chemmater.2c00288 | yes | PMC9202302 full text |
| K2Mn[Fe(CN)6] ambient structure | P21/n, a = 10.0912, b = 7.3243, c = 6.9442 Å, β = 90.02° | Å | K1.94Mn[Fe(CN)6]0.994 | 10.1038/s41467-021-22499-0 | yes | PMC8041879 |
| Monoclinic → cubic on K extraction (electrochemical, not thermal) | x ≈ 0.45 | – | K2-xMn[Fe(CN)6] | arXiv:2509.04587 | yes | arXiv PDF |
| RbMnFe charge-transfer transition (Fe(II)Mn(III) ↔ Fe(III)Mn(II)) | T1/2↓ = 240 K, T1/2↑ = 297 K (hysteresis 57 K) | K | Rb0.97Mn[Fe(CN)6]0.98·1.03H2O | arXiv:0810.0175 (Lummen et al.) | yes | arXiv PDF |
| Prussian blue ferromagnetic Tc | 5.6 | K | Fe4[Fe(CN)6]3·xH2O | Comptes Rendus Chimie 22 (2019) 435, doi 10.1016/j.crci.2019.07.004; primary: Herren et al. Inorg. Chem. 1980, 10.1021/ic50206a032 | yes (secondary review) | Comptes Rendus page; primary abstract not read |
| Na2MnFe(CN)6 hydrated/dehydrated phases (Song 2015) | abstract gives no transition T | – | – | 10.1021/ja512383b | abstract read; no T found | PubMed |

## 4. C≡N and low-frequency vibrations

| quantity | value | units | conditions | source DOI | verified | how |
|---|---|---|---|---|---|---|
| IR ν(CN), Prussian blue KFe(III)[Fe(II)(CN)6]·~10H2O | 2070 (strong, broad) | cm-1 | FT-IR/ATR, powder | Kulesza et al., Anal. Chem. 1996, 68, 2442 (DOI not checked) | yes | PDF text |
| IR ν(CN), Berlin green / oxidised PB | new band at 2169 (difference spectrum) and 2163; narrow band at 2074 | cm-1 | same | same | yes | same |
| IR ν(CN), Everitt's salt / PW K2Fe(II)[Fe(II)(CN)6] | single band near 2060–2070 (difference peak at 2057) | cm-1 | same | same | yes | same |
| IR ν(CN), Prussian white K2Fe[Fe(CN)6] | ~2067 (single; assigned to Fe2+ only) | cm-1 | low-water KFHCF powder | 10.22146/ijc.99310 | yes | PDF text |
| Ranges by linkage (table) | Fe(II)-CN-M(II) 2065–2100; Fe(II)-CN-M(III) 2090–2140; Fe(III)-CN-M(II) 2146–2185; Fe(III)-CN-M(III) 2180–2210; Fe(II)-CN-Mn(II) 2065; Fe(III)-CN-Mn(II) 2146–2170; CN-(aq) 2080 | cm-1 | compilation | arXiv:0810.0175, Table II | yes | arXiv PDF |
| Ferro- and ferricyanide in solution | [Fe(CN)6]4- 2037; [Fe(CN)6]3- 2116 | cm-1 | aqueous, FTIR | arXiv:1501.01439 | yes | arXiv PDF |
| Solid K4Fe(CN)6 ~2044 / K3Fe(CN)6 ~2118 | not found in a source I read | cm-1 | – | – | no | gap |
| Raman ν(CN), FeHCF film (PB) | 2094.7, 2157.3 | cm-1 | electrodeposited in KCl | arXiv:2609.26008, Table 2 | yes | arXiv PDF |
| Raman, Prussian blue (Moretti & Gervais) | 2152, 2091 (CN); 510 (Fe–C stretch); 276 (Fe–C≡N–Fe deformation) | cm-1 | pigment | 10.1002/jrs.5366 | secondary | quoted in arXiv:2601.03313, Table 3 |
| Raman Fe–C region, PB | bands near 598, 528, 509, 445; 528/509 intensity ratio flips between "soluble" and "insoluble" PB | cm-1 | – | 10.1016/j.jelechem.2011.06.022 | secondary | quoted in arXiv:2601.03313 |
| Far-IR Fe–C–N–Fe bend | 450–650 region | cm-1 | – | Kulesza 1996 | yes (range only) | PDF |

## 5. Compressibility and pressure

| quantity | value | units | conditions | source DOI | verified | how |
|---|---|---|---|---|---|---|
| B0, MnPt(CN)6 (stoichiometric) / ·nH2O | 33(2) / 35(2); pT = 0.97(12) / 1.31(10) GPa | GPa | XRD, BM2 | 10.1021/jacs.0c13181 | yes | PMC8028041 full text |
| B0, defective Mn[Co(CN)6]0.67, dehydrated / hydrated | 6.5(7) / 13.5(6); hydrated pT = 1.46(13) GPa | GPa | – | same | yes | same |
| B0, Cs1Mn[Co(CN)6] | 31(2); → P-4n2 at 1.94(15) GPa (B0 of that phase 14.7(8)) | GPa | – | same | yes | same |
| B0, Rb0.87Mn[Co(CN)6]0.91 | ≈20 (from 2 points); transition at 0.23(5) GPa | GPa | neutron | same | yes | same |
| B0, FePt(CN)6 | 33(5) | GPa | – | same | yes | same |
| B0, Cu[Co(CN)6]0.67 hydrated / dehydrated | 35.2(11) / 14.2(4); dehydrated form amorphises at about 0.7 GPa (below 1 GPa) | GPa | – | 10.1039/d3sc06912e | yes | PMC10901509 |
| B0, CsCuCo / RbCuCo | 42.4(2) X-ray / 23.66(15) X-ray; RbCuCo → P2/m at 2.9(2) GPa | GPa | – | same | yes | same |
| Amorphisation of FeCo(CN)6 | ~10 GPa (and decomposition by 17.4 GPa) | GPa | – | cited in 10.1039/d3sc06912e and 10.1038/s41467-022-31658-w (PRB 77, 064104) | secondary | – |
| DFT K, Fe[Fe(CN)6]·2H2O / Fe[Fe(CN)6] / Fe[Fe(CN)6]3/4·3.5H2O | 114.7 / 111.6 / 37.3 (VRH) | GPa | DFT (theory, not experiment) | 10.1038/s41467-022-31658-w | yes | Nat. Commun. PDF |
| Experimental B0 for PB Fe4[Fe(CN)6]3 or any K-PBA | **not found**; high-pressure Raman of PB exists (Barsan et al., 10.1002/jrs.2931; 511 cm-1 Fe–C at ambient), but I did not read it | – | – | – | no | gap |

## Notes for the MD validation

- The DFT bulk modulus of vacancy-free FeFe(CN)6 (~112 GPa) is about 3× the experimental B0 of stoichiometric cubic PBAs (31–42 GPa). Experimental B0 also drops sharply with vacancies (6.5 GPa) and depends on water. Comparisons should be made at matched vacancy and water content.
- In the A-free framework, α is negative (−4.26e-6/K). Na insertion pushes it positive (+4.0e-6/K at Na1.5). K-rich frameworks should therefore expand normally (PTE), but I found no measured value for them.
- The A-free lattice constant is 10.13182(4) Å (Gao 2022). Ito et al. quote 10.191 Å as the experimental value for PB.
- Sites were blocked by bot protection (HTTP 403), not by a login: pubs.rsc.org HTML, the MDPI full text, and DOAJ article pages. They are not added to BLOCKED.txt because no login or email was asked for.

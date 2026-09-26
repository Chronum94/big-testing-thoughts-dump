# Factorized species-pair radial functions for GMNN

Working notes → paper draft. Branch `factorized-radial` in `~/apax` (uncommitted as of 2026-09-26).

---

## 1. Motivation

GMNN's `RadialFunction` (`apax/layers/descriptor/basis_functions.py`) contracts `n_basis` radial basis functions
into `n_radial` channels with a **dense per-species-pair coefficient table**:

```
embeddings: (n_species=119, n_species=119, n_radial, n_basis)     # indexed [Z_centre, Z_nbr]
radial_r(r; Zc, Zn) = Σ_b W[Zc, Zn, r, b] · basis_b(r) · cutoff(r)
```

This is the original GM formulation (Zaverkin & Kästner 2020, 2021: β_{Z_i,Z_j,s,k}). Every pair is independent:
learning (Zi, Zj) says nothing about (Zi', Zj) or (Zi', Zj'), even though chemistry (periodic groups, similar
radii/valence) says it should. Unseen pairs stay at random init. Parameter count scales as n_species².

## 2. Literature

### 2.1 Element-embedding / compression lineage
- **Willatt, Musil, Ceriotti 2018** (arXiv:1807.00236) — alchemical compression: element-resolved densities
  replaced by a few learned "pseudo-element" channels (one weight vector per element). Learned element vectors
  cluster in a way reminiscent of the periodic table.
- **Lopanitsyna et al. 2023** (PRMaterials 7, 045802; HEA25) — alchemical compression to 25 d-block metals; big cost
  reduction, negligible accuracy loss, stable extrapolation. Also compresses element + radial indices jointly.
- **Darby, Kovács, Csányi et al. 2023** (PRL 131, 028001; TRACE) — element channels in ACE-type descriptors as a
  tensor to factorize; representation size independent of number of elements.
- **SpookyNet** (Unke et al. 2021) — element embedding initialised from ground-state electron configuration, plus a
  learned per-element residual; explicit "alchemical" inductive bias.
- **DPA-1** — learned type embeddings arrange in a spiral matching the periodic table (periods along the spiral,
  groups across it).

### 2.2 What frontier universal models do (none keep a pair table)

| Model | Per-element | How pair dependence enters | Physical prior |
|---|---|---|---|
| MACE (MP-0, OFF, OMAT) | one-hot → linear | radial MLP sees distance only; species via sender features × R(r); centre-element-indexed symmetric-contraction weights | optional Agnesi transform (covalent radii), ZBL |
| NequIP / SevenNet | one-hot → linear | same as MACE | no |
| Allegro | one-hot Zi, Zj | concat(onehot_i, onehot_j, basis(r)) → two-body MLP | no |
| EquiformerV2 / eSEN / UMA | nn.Embedding | radial MLP input concat(RBF(r), src_emb(Zj), tgt_emb(Zi)) | UMA: experts mixed by composition/charge/spin/task |
| Orb-v3 | atomic-number embedding | Bessel(8) ⊗ SH(L≤3) edges; species via nodes | no |
| PET / PET-MAD | nn.Embedding (centre, nbr) | edge token = geometry + nbr-species embedding; transformer | no |
| MatterSim / M3GNet / CHGNet | nn.Embedding | species-agnostic RBF edges; via nodes | no |
| GRACE | 128-d chemical embedding | tensor decomposition of ACE coefficients | no |
| DPA-1/2/3 | learned type embedding | embedding net on s(r) ⊕ type embeddings | optional `use_econf_tebd` |
| SpookyNet | e-config → linear + residual | via nodes | yes |

Patterns: (a) multiplicative species-agnostic R(r) × neighbour features, or (b) concat (r, e_i, e_j) → MLP.
Tables survive only for the **centre** element (E0s, scale/shift, MACE contraction weights): O(n_species).
Embeddings are almost always randomly initialised — frontier models have ~10⁸ structures to learn the periodic
table; physics priors (e-config, radii, ZBL) are rare and matter most at small-data scale.
(MACE contraction-weight indexing and Agnesi details stated from memory, not re-verified against source.)

### 2.3 Closest relatives of our model
Our model:
```
radial_r(r; Zc, Zn) = norm · Σ_k u_k(Zc) · v_k(Zn) · (C_k · basis(r))_r
```
- **Alchemical compression / TRACE**: `v_k(Zn)·(C_k·basis)` is exactly a pseudo-element channel with its own radial.
- **MACE layer-1 A-basis**: A_{i,k} = Σ_j W_{k,Zj} R_k(r_ij) Y — same structure, but MACE keeps k as a feature
  channel, uses an MLP radial, and adds the centre element later.
- **What's specific to ours**: an explicit **centre factor u_k(Zc)** inside the radial → full **CP (canonical polyadic, a.k.a. CANDECOMP/PARAFAC) tensor decomposition** of
  the (Zc, Zn, basis, radial) coefficient tensor; and **collapse over k** into n_radial outputs (rank decoupled from
  output width).
- **Cross-field**: factorization machines (Rendle 2010) / low-rank matrix completion — bilinear pair interaction of
  learned vectors → generalises to "cold" (unseen) pairs; also fails for never-seen *elements* without a prior.
- **Physics analogy**: Lorentz–Berthelot ε_ij = √(ε_i ε_j) is a rank-1 multiplicative factorization of a pair
  parameter; ours is a learned, rank-k, radially resolved generalisation. (Arithmetic σ_ij = (σ_i+σ_j)/2 is additive
  — closer to apax's covalent radial transform.)
- **Not like**: full pair tables (GMNN, MTP, SOAP/GAP per-species densities); nonlinear pair MLPs (Allegro, DeePMD,
  EquiformerV2) — more capacity, no strict low-rank structure, more per-pair compute.

### 2.4 Why MAD has dimers/trimers ("condition the low-body-order expansion"; our interpretation)
E = Σ E1 + Σ E2 + Σ E3 + … . Bulk data only constrains the sum → the split between body orders is
non-identifiable (ill-conditioned), yet low orders govern dissociation, low coordination, short-range repulsion.
Dimers (E1+E2) and trimers (E1+E2+E3) pin them; `monomers` pin E1.
For us: we trained on exactly the subset that probes the pair/3-body channel (hence n_contr 4 ≈ 8). Dimer data is
sparse per pair (4.7k of ~7k possible pairs at 10k frames) → **factorisation makes the conditioning data propagate
to pairs without dimer frames** (paper framing). GMNN's nonlinear readout isn't strictly body-ordered, so dimer
data shapes the low-coordination region of the MLP rather than a separable E2.

## 3. Method

### 3.1 FactorizedRadialFunction
```python
u = pair_emb_centre   # (n_species, rank)
v = pair_emb_nbr      # (n_species, rank)
C = pair_core         # (n_basis, rank * n_radial)   stored flat → one dense GEMM

basis  = basis_fn(radial_transform(dr))                    # (P, n_basis)
proj   = (basis @ C).reshape(P, rank, n_radial)             # dense GEMM, no gather, precision=HIGHEST
w      = u[Z_centre] * v[Z_nbr]                             # (P, rank) — only per-pair gather: 2·rank floats
radial = norm * einsum("pk,pkr->pr", w, proj) * cutoff(dr)  # precision=HIGHEST
```
Same `__call__(dr, Z_i, Z_j)` signature as `RadialFunction`; `Z_j` = centre (matches `embeddings[Z_j, Z_i]`).
Config: GMNN `radial_rank: Optional[int]` (None → dense table), `radial_emb_jitter: float = 0.1`.
Optimizer groups `pair_emb_centre`, `pair_emb_nbr`, `pair_core` → `emb_lr`.
`radial_factor_mode`: `cp` (u·v, default), `nbr` (v(Zn) only), `centre` (u(Zc) only); only used embeddings are created.
Shapes: u, v = (n_species=119, rank); C = (n_basis, rank·n_radial). "CP" = canonical polyadic decomposition.

### 3.2 Initialisation
Single init path (any basis): u, v = 1 + jitter·N(0,1); C = ½ + √rank·U[-½,½]; norm = 1/rank.
(Early etoh runs used an extra signed branch — u, v ~ N(0,1), C ~ U[-1,1], norm 1/√(n_basis·rank) — for
non-one-sided bases; removed as unneeded.)
  → entry mean/var match U[0,1]; at jitter=0 every pair is identical (= mean_k C_k); pair-to-pair spread ∝ jitter.

Measured at init (3 elements, rank 9, 8 Bessel, 5 radial):

| | entry mean | entry std | between-pair std |
|---|---|---|---|
| dense U[0,1] | 0.49 | 0.29 | 0.27 |
| jitter 0.1 | 0.43 | 0.28 | 0.03 |
| jitter 1.0 | 0.46 | 0.51 | 0.32 |

Interpretation: jitter sets how far each pair starts from a **shared radial function**. Small jitter = shrinkage
prior toward "one radial + small element-specific corrections".

### 3.3 How it conditions Zi' on Zi
Fix partner Zj: W[·, Zj] = u(·)·G(Zj) with G(Zj) = v(Zj)∘C (rank × n_radial·n_basis). All partners of Zj live in a
shared rank-d subspace; pairs (Zi, Zj), (Zi'', Zj) fit G(Zj); Zi' needs only its d numbers u(Zi'), learnable from Zi'
paired with *any* Zk → matrix-completion prediction of W[Zi', Zj]. Per-element cost: 84/partner → d total.
Limits: (1) an element absent from data gets zero gradient; (2) nothing pulls u(Zi') toward u(Zi) unless data says
so; (3) separate u/v decouple centre and neighbour roles (symmetric u=v couples them).
Prior (not yet tested): u(Z) = φ(Z)·A + δ(Z), φ = fixed element features (group/period one-hot, e-config, χ, r_cov),
δ zero-init + weight decay.

## 4. Cost analysis (GPU memory hierarchy)

### 4.1 Default GMNN (n_basis 7, n_radial 5, readout [256,256], fp32); P = pairs, N = atoms; ref 10k atoms × 50 nbrs

| Tensor | Size | Fits |
|---|---|---|
| species-pair table 119²·5·7 | 1.98 MB | L2 |
| … at n_radial=n_basis=16 | 14.5 MB | L2 on A100/H100/4090; not 3090 (6 MB) |
| used pair slices, 4 elements | 2.2 KB | L1 |
| gathered coeffs P×35 | 140 B/pair → 70 MB | HBM |
| per-pair moments P×5×40 | 800 B/pair → 400 MB if materialised | HBM |
| per-atom moments | 800 B/atom → 8 MB | L2 |
| GM features N×360 | 1.4 KB/atom | L2 |
| readout weights | ~630 KB | L2 |

### 4.2 Proposed config (n_basis 12, n_radial 7, readout [64, 8])

| Tensor | Size | Fits |
|---|---|---|
| pair table 119²·7·12 | 4.76 MB | L2 |
| gathered coeffs | 336 B/pair → 168 MB | HBM |
| per-pair moments P×7×40 | 1.12 KB/pair → 560 MB (280 MB symmetric) | HBM |
| per-atom moments | 1.12 KB/atom → 11 MB | L2 |
| GM features N×**910** | 3.6 KB/atom → 36 MB | L2 H100/4090; tight A100; not 3090 |
| readout W1 910×64 | 233 KB | just over H100 smem (228 KB); fits in bf16 |

Feature count n_radial=7: contr0 7, contr1–3 28 each, contr4 84, contr5/6 196 each, contr7 343 = 910.

Findings:
- The **table itself is never the problem** (only k² slices read, cache-resident). Costs are the **materialised
  per-pair intermediates**: gathered coeffs (35× amplification of a 4-byte distance) and per-pair moments. XLA
  likely doesn't fuse gather into the batched dot; moment fusion into segment_sum unverified.
- Per-pair traffic ≈ (336 + 1120)·50 ≈ **73 KB/atom** vs 3.6 KB/atom features; ×2 for forces.
- Symmetric moment storage: 40 → 20 components halves per-pair and per-atom moment traffic.
- W1 910→64 is ~32 FLOP/B: near fp32 CUDA-core ridge, memory-bound under TF32; features must be materialised
  (concat + tril gathers) before cuBLAS.
- Readout ~117k FLOP/atom (almost all W1) vs descriptor contractions ~40k + per-pair ~18k.
- Fused kernel: 280 moment floats/pair > 255-register limit → split across threads (e.g. per radial channel).
- Factorised embedding at d=8: element vectors 3.8 KB, core d²·7·12 = 21.5 KB → whole thing fits in smem.

### 4.3 Gathers: where HBM actually gets hit
- Gather *sources* are small (pair table k²×84 ≈ 5 KB; per-atom embeddings N×d ≈ 480 KB at 10k×12) → L1/L2.
- HBM cost = **materialised gather outputs** (P×width) + index arrays (8 B/pair). A fusion question.
- Environment-aware embeddings (CMNN-style): real cost is an **extra full neighbour pass** (segment_sum must finish
  first), position-dependent coeffs (no precompute, fp32 forced, backprop through the embedding path); gather
  traffic ~2d·4 B ≈ 96 B/pair is modest.
- Fixed embeddings: expanding to a full table then gathering still costs 84 floats/pair. CP form gathers only 2·rank
  floats (32 B at rank 4 vs 336 B) + one dense GEMM.

### 4.4 Rank scaling of per-pair traffic

| | (8 basis, 5 radial) | (12, 7) |
|---|---|---|
| params at rank 64 | 17,792 (71 KB) | 20,608 (82 KB) |
| gather u,v at rank 64 | 512 B | 512 B |
| dense-table gather | 160 B | 336 B |
| `proj` at rank 64 | 1.3 KB | 1.8 KB |
| gather / `proj` at rank 8 | 64 B / 160 B | 64 B / 224 B |

Bandwidth win only at low rank. For high rank, contract u⊙v with C first (per-pair weights n_basis×n_radial) → cost
equals dense regardless of rank.

### 4.5 Arithmetic intensity, CP vs dense (per pair, fp32, (12, 7); estimates)

| Variant | FLOPs/pair | HBM B/pair unfused | FLOP/B unfused | FLOP/B fused (40 B) |
|---|---|---|---|---|
| dense (gather 84, contract) | ~170 | ~800 | ~0.2 | ~4 |
| CP rank 8 | ~1,460 | ~710 | ~2 | ~37 |
| nbr-only rank 8 | ~1,450 | ~580 | ~2.5 | ~36 |
| CP rank 64 | ~11,600 | ~4,700 | ~2.5 | ~290 |

- Unfused (XLA today): all memory-bound; CP rank 8 slightly fewer bytes than dense; rank 64 ~6× more.
- Fused MD kernel: u, v, C (few KB) stay in smem/registers; cost becomes FLOPs. A per-step/cached table
  W[a,b] for present species (k²·84 floats, ~5 KB at k=4) also fits smem and costs dense FLOPs (~170) → ~9×
  cheaper than CP on-the-fly at rank 8. Static embedding ⇒ materialise the table at inference: dense runtime
  with CP generalisation. (Optional `radial_materialize` path — not implemented.)

## 5. Precision analysis

Requirement: E and F smooth in positions; distances + basis in fp32 (reference resolution); energy changes ~1e-7.
- **Weights ≠ activations.** Rounded weights are constants → still smooth, just a different function. Rounding any
  position-dependent activation to ε makes E piecewise constant (steps ~ε·|x|·|∂E/∂x|); autodiff treats rounding as
  identity → forces ≠ −∇E → NVE drift. bf16 ε≈3.9e-3, tf32/fp16 ε≈4.9e-4: 3–4 orders above 1e-7.
- ⇒ everything on the positions→energy path stays fp32; scale/shift + energy sum fp64 (already: `fp64_sum`,
  `models.py:117,122`). Exception: branch scaled by s tolerates ε ≲ 1e-7/s.

| Component | Weights | Compute |
|---|---|---|
| positions → dr_vec | – | ≥fp32; subtract in fp64 for large cells (50 Å fp32 → 6e-6 Å) |
| neighbour list build | – | fp16/bf16 OK (discrete; cutoff smooth; error ≪ skin) |
| basis | – | fp32 |
| pair table | bf16 storage OK, upcast | only weight replicated per pair → halves gather traffic |
| radial, moments, segment_sum | – | fp32 |
| GM contractions | – | fp32, precision=HIGHEST |
| readout W1 | bf16 storage OK | fp32 accumulate; or bf16×3 / tf32×3 for tensor cores |
| hidden, W2/W3 | bf16 OK | fp32 |
| scale/shift, sum | – | fp64 |
| uncertainty heads | bf16 OK | bf16 probably OK |
| optimizer state | fp32 master | bf16 Adam moments possible |

- Low-precision weights buy little here: weights already sit in L2; traffic is activations, which can't be lowered.
- **TF32 hazard**: nothing in `apax/` sets `jax_default_matmul_precision` or `precision=`. On Ampere+ JAX DEFAULT
  precision allows TF32 for f32 dot_general → descriptor einsums/readout at ~5e-4 input precision. Unverified here
  (CPU-only jaxlib). Fix: `jax.config.update("jax_default_matmul_precision", "highest")` or per-op `precision`.
- Factorised table is position-independent → a weight transform, no smoothness/precision issue; can be cached.

## 6. Experiments

Env: `apaxenv`, CPU (`JAX_PLATFORMS=cpu`), taskset-pinned. Bessel basis n_basis 8, n_radial 5, readout [64,32,16],
elu readout activation, adam lr 1e-3, cyclic_cosine schedule (period = n_epochs).

### 6.1 Ethanol (3 elements, 1496/155, 40 epochs, single seed)
Config base: `config_ab_scaledelta_spm1.yaml` with `softplus_m1`→`elu`, `self_attention` dropped (not on branch);
covalent radial transform, NLH correction. Configs `config_ab_fr_*.yaml`.

| Run | val E (meV) | val F (meV/Å) |
|---|---|---|
| dense | 25.4 | **42.3** |
| rank 2 (signed-normal init) | 57.2 | 72.9 |
| rank 4 | 52.5 | 66.2 |
| rank 9 | 26.6 | 53.0 |
| rank 9, moment-matched init (jitter 0.1) | 33.6 | 51.9 |
| rank 9, jitter 1.0 | **20.8** | 45.1 |

- Capacity monotone in rank (73→66→53). Rank 9 ≥ dense expressiveness for 3 species (indicator embeddings), yet
  gap remained → init. Pair diversity at init (jitter 1.0) closed most of it (11 → 3 meV/Å, likely within noise).
- Data-rich pairs → pairs need to differ → large jitter better here (opposite of MAD, §6.5).
- **3-seed check, current code (rank 8, jitter 0.1, 40 epochs)**: dense val E 25.2±0.9 meV, val F 44.4±1.8 meV/Å
  (train F 38.9±3.3); rank 8 val E 39.6±8.2, val F 60.5±9.9 (train F 54.1±5.5; per-seed F 50.2/70.0/61.2).
  Factorised is ~36% worse on forces and much noisier across seeds on data-rich few-element sets; train F also
  higher → underfitting/slow optimisation, not overfitting. Small-jitter shrinkage prior is the wrong prior here.

### 6.2 MAD-1.6 dimers + trimers
- Source `mad-1.6-r2scan-train.xyz` subsets: binary_random 58867, dimers 31325, madcat 42852, mc 78134, monomers 102,
  shiftml_molcrys 6857, shiftml_molfrags 2593, trimers 2306, trimers-extended 92278.
- Filter: `subset ∈ {dimers, trimers}` (33,631 frames), random sample (seed 0) via `mad_dimtri_subset.py N`.
  - 2.5k: 102 elements, 2,266 element pairs, pair-count quartiles 1/1/2, max 82.
  - 5k (`mad_dimtri_5000.traj`, superset of 2.5k): 3,568 pairs, quartiles 1/1/2, max 137.
- Energies are all-electron totals (~−7e5 eV for heavy dimers). |F| p50/p90/p99/max = 1.9/32.5/90/100 eV/Å;
  min distances p1/p50/p99 = 1.16/2.74/4.92 Å.
- Config: per_element_regression_shift (λ=1e-6), per_element_force_rms_scale, `radial_transform: identity`,
  no empirical corrections (so all element info goes through the radial coefficients), batch 16, 200 epochs.
- Metrics: MAE, E per structure (eV), F per component (eV/Å); val split into frames whose element pairs all appear
  in train ("seen") vs ≥1 unseen pair. `ASECalculator` loads the **best-val checkpoint** (dense = effectively
  early-stopped; favours dense).

**Gotcha**: apax `per_element_regression_shift` default ridge `energy_regularisation: 1.0` shrinks per-element
references toward the global mean; with all-electron energies (element means differ by ~1e5 eV, ~50 frames/element)
this gave ~8 keV residual MAE (train E MAE 2.3 keV, loss ~1e8). λ=1e-6 → 3.1 eV. Worth a code-level warning/fix.

### 6.3 MAD 2.5k (2250/250; 80 seen / 170 unseen-pair val frames; single seed)

| Model | train F (log, ep199) | seen E / F | unseen E / F |
|---|---|---|---|
| linear + 0F | – | 2.75 / 1.92 | 3.06 / 4.26 |
| dense | 0.02 | 2.26 / 1.35 | **3.33** / 2.81 |
| rank 8 (j1.0) | 0.32 | 2.19 / 1.08 | 2.49 / 1.99 |
| rank 16 | 0.27 | 2.06 / 1.07 | 2.35 / 2.18 |
| rank 32 | 0.21 | 2.04 / 1.56 | 2.29 / 1.88 |

Dense memorises; unseen-pair energy worse than linear baseline. Timing (4 cores, 200 ep): ~0.53–0.55 s/epoch all.

### 6.4 MAD 5k (4500/500), 5 seeds (seed also changes split), jitter 1.0
Val frames: seen/unseen = 280/220, 310/190, 291/209, 285/215, 310/190. `eval_mad5k.py`.

| Model | seen E | seen F | unseen E | unseen F |
|---|---|---|---|---|
| linear + 0F | 3.13±0.21 | 3.45±0.30 | 3.18±0.14 | 4.09±0.43 |
| dense | 2.49±0.11 | 2.25±0.13 | 3.05±0.10 | 3.08±0.20 |
| rank 8 | 1.99±0.14 | 1.23±0.09 | 2.07±0.14 | 1.51±0.25 |
| rank 16 | 1.95±0.09 | 1.27±0.07 | 2.06±0.12 | 1.54±0.10 |
| rank 32 | 1.92±0.09 | 1.24±0.09 | 2.12±0.10 | 1.62±0.14 |

Log-metric (final epoch, seeds 1/2): dense val F 1.99/1.89, rank 8 0.90/0.89 (apax log metric differs from ours).

Timing (2 cores, 8 concurrent runs): dense 2.19 s/epoch median (457 s total), rank 8/16/32 1.80–1.87 (~400 s);
first epoch ~20 s (JIT). Factorised ~15–18% faster — dense-table gradient scatter + Adam state on 566k entries.
Not representative of GPU forward cost (2–3 atom systems).

Slope over last 30 epochs (% per 100 epochs): dense train F −86…−93%, val F +0.6…+1.5% (overfitting);
rank 8 train F −5.5…−6.7%, val F −0.5…+1.0% (flat, mixed sign). **But** LR has annealed to ~0 (period = n_epochs), so
flatness is the schedule, not convergence. Test: longer schedule (600 ep) — not yet run.

### 6.5 Dense + weight decay scan (seed 1 split; adamw, decoupled)
Decoupled decay shrinks by exp(−Σ lr·wd) ≈ exp(−28·wd) over this run: 1e-5 → 0.03%, 1e-3 → 2.8%, 1e-2 → 24%,
1e-1 → 94%. `eval_wd_s1.py`.

| Model | train E | train F | seen E | seen F | unseen E | unseen F |
|---|---|---|---|---|---|---|
| linear + 0F | 3.08 | 3.63 | 3.24 | 3.77 | 3.05 | 3.84 |
| dense | 1.28 | 0.41 | 2.39 | 2.33 | 3.06 | 2.81 |
| dense wd 1e-5 | 1.28 | 0.41 | 2.39 | 2.33 | 3.06 | 2.81 |
| dense wd 1e-4 | 1.22 | 0.41 | 2.43 | 2.38 | 3.06 | 2.86 |
| dense wd 1e-3 | 1.49 | 0.46 | 2.44 | 2.35 | 3.09 | 2.87 |
| dense wd 1e-2 | 1.68 | 0.51 | 2.45 | 2.30 | 3.17 | 2.89 |
| dense wd 1e-1 | 2.26 | 0.80 | 2.61 | 2.18 | 3.37 | 2.68 |
| rank 8 (j1.0) | 1.71 | 0.49 | 2.04 | 1.37 | 1.96 | 1.21 |
| rank 16 | 1.64 | 0.52 | 1.97 | 1.19 | 2.19 | 1.50 |
| rank 32 | 1.51 | 0.37 | 1.89 | 1.24 | 2.05 | 1.50 |

- Weight decay doesn't rescue dense; strong wd trades train fit for ~nothing.
- Train→unseen F ratio: dense 6.9×, rank 8 2.5×. Factorised train error ≈ dense → the gain is **cross-pair sharing,
  not regularisation**. Weight decay can shrink a table; it can't make it share.

### 6.6 Jitter scan (rank 8, MAD 5k)
Seed 1 (`eval_wd_s1.py dense r8j0.1 r8j0.316 r8`):

| Model | train E | train F | seen E | seen F | unseen E | unseen F |
|---|---|---|---|---|---|---|
| dense | 1.28 | 0.41 | 2.39 | 2.33 | 3.06 | 2.81 |
| j1.0 | 1.71 | 0.49 | 2.04 | 1.37 | 1.96 | 1.21 |
| j0.316 | 1.43 | 0.35 | 1.65 | 0.68 | 1.52 | 0.72 |
| j0.1 | 1.41 | 0.34 | 1.58 | 0.57 | 1.52 | 0.69 |

3 seeds (`eval_multi.py 1,2,3 dense r8 r8j0.1 r8j0.01`):

| Model | train E | train F | seen E | seen F | unseen E | unseen F |
|---|---|---|---|---|---|---|
| dense | 1.37±0.09 | 0.44±0.02 | 2.43±0.08 | 2.20±0.15 | 2.98±0.07 | 2.98±0.15 |
| rank 8 j1.0 | 1.69±0.03 | 0.49±0.01 | 2.00±0.16 | 1.25±0.11 | 1.99±0.13 | 1.34±0.13 |
| **rank 8 j0.1** | 1.44±0.02 | **0.37±0.02** | **1.58±0.01** | **0.59±0.01** | **1.57±0.05** | **0.73±0.08** |
| rank 8 j0.01 | 1.44±0.03 | 0.38±0.02 | 1.59±0.02 | 0.61±0.02 | 1.57±0.09 | 0.75±0.11 |

- Small jitter wins on every metric **including train** (optimises better, not just generalises better).
- ≤0.1 saturates. Default stays 0.1.
- Headline: factorised + small jitter ≈ **4× lower unseen-pair force error than dense** (0.73 vs 2.98), ~3.7× on
  seen pairs; energy below linear baseline by ~1.6 eV where dense barely beats it.
- Dataset-dependent optimum: sparse pairs → shrink toward shared radial (small jitter); data-rich few-element
  (ethanol) → pairs need diversity (jitter 1.0 won at 40 epochs).

### 6.7 Rank 64, 10k, batch size, force bins (all seed 1, jitter 0.1; exploratory)
Rank 64 on 5k: train F 0.32, seen F 0.59, unseen E/F 1.46/0.78 (rank 8: 0.34, 0.57, 1.52/0.69) → no gain.

10k subset (`mad_dimtri_10000.traj`, 9000/1000; 4,708 pairs, quartiles 1/2/3, max 267):

| Model | train E | train F | seen E | seen F | unseen E | unseen F |
|---|---|---|---|---|---|---|
| dense | 1.79 | 0.65 | 2.46 | 2.27 | 2.79 | 2.89 |
| rank 8 | 1.26 | 0.31 | 1.33 | 0.47 | 1.26 | 0.51 |
| rank 64 | 1.01 | 0.21 | 1.08 | 0.48 | 0.95 | 0.52 |

5k→10k unseen F: rank 8 0.69→0.51, rank 64 0.78→0.52, dense 2.81→2.89. Factorised still data-limited; rank 64
converts capacity into energy at 10k.

Per-atom |ΔF| (eV/Å) by |F_ref| bin, 10k val (`eval_fbins.py`):

| Model | median | 0–1 (n=666) | 1–5 (739) | 5–20 (315) | 20–50 (207) | 50+ (141) |
|---|---|---|---|---|---|---|
| dense | 1.95 | 1.53 | 2.45 | 8.63 | 14.9 | 34.4 |
| rank 8 | 0.78 | 0.69 | 0.78 | 1.65 | 2.31 | 5.00 |
| rank 64 | 0.79 | 0.73 | 0.80 | 1.56 | 2.57 | 4.67 |

Win holds in every bin; low-force error ~ force magnitude for all models (MSE dominated by high-force frames).

Batch 8 vs 16 (5k): forces within noise; rank 64 unseen E 1.46→1.31.

Cost, core-s/epoch (median s/epoch × pinned cores; crude, sublinear core scaling, varying concurrency):
5k bs16: dense 4.4, r8 (j1.0) 3.6, r8 (j0.1) 2.6, r64 4.1. 5k bs8: 9.6–9.9. 10k bs16: 10.2–11.0.
bs8 costs ~2–3× bs16 per epoch; rank barely matters.

### 6.8 n_contr 4 (drop 3-moment / 4-body contractions), bs16, 3 seeds, rank 8 j0.1
2.5k = `mad_dimtri.traj` (2250/250), 5k as before.

| Set | Model | train E | train F | seen E | seen F | unseen E | unseen F |
|---|---|---|---|---|---|---|---|
| 2.5k | dense | 1.44±0.14 | 0.37±0.02 | 2.47±0.30 | 1.87±0.51 | 3.31±0.20 | 3.03±0.32 |
| 2.5k | dense nc4 | 1.49±0.27 | 0.34±0.03 | 2.50±0.32 | 1.92±0.53 | 3.22±0.20 | 3.01±0.30 |
| 2.5k | rank 8 | 1.71±0.09 | 0.60±0.19 | 1.84±0.16 | 0.68±0.14 | 1.86±0.20 | 1.04±0.12 |
| 2.5k | rank 8 nc4 | 1.73±0.06 | 0.63±0.04 | 1.87±0.12 | 0.74±0.14 | 1.89±0.17 | 1.05±0.12 |
| 5k | dense | 1.37±0.09 | 0.44±0.02 | 2.43±0.08 | 2.20±0.15 | 2.98±0.07 | 2.98±0.15 |
| 5k | dense nc4 | 1.37±0.10 | 0.44±0.01 | 2.42±0.09 | 2.19±0.08 | 3.00±0.08 | 2.91±0.10 |
| 5k | rank 8 | 1.44±0.02 | 0.37±0.02 | 1.58±0.01 | 0.59±0.01 | 1.57±0.05 | 0.73±0.08 |
| 5k | rank 8 nc4 | 1.46±0.03 | 0.38±0.02 | 1.59±0.03 | 0.57±0.02 | 1.60±0.09 | 0.71±0.08 |

- n_contr 4 ≈ 8 everywhere (within 1σ): 3-moment contractions carry no information on dimers/trimers, and the
  factorised advantage doesn't depend on them. Dense still memorises with 50 features → overfitting lives in the
  pair table, not readout capacity on redundant features.
- Factorised gain grows with data: unseen F rank 8 1.04 (2.5k) → 0.73 (5k) → 0.51 (10k, 1 seed); dense flat ~3.0.

### 6.9 Factor ablation (5k, rank 8, j0.1, n_contr 8, bs16, 3 seeds; `radial_factor_mode`)

| Model | train E | train F | seen E | seen F | unseen E | unseen F |
|---|---|---|---|---|---|---|
| dense | 1.37±0.09 | 0.44±0.02 | 2.43±0.08 | 2.20±0.15 | 2.98±0.07 | 2.98±0.15 |
| CP u·v | 1.44±0.02 | 0.37±0.02 | 1.58±0.01 | 0.59±0.01 | 1.57±0.05 | 0.73±0.08 |
| nbr only v(Zn) | 1.53±0.03 | 0.52±0.04 | 1.66±0.04 | 0.60±0.04 | 1.58±0.10 | 0.78±0.07 |
| centre only u(Zc) | 2.10±0.08 | 0.97±0.08 | 2.18±0.13 | 1.12±0.14 | 2.24±0.09 | 1.35±0.11 |

- v-only ≈ CP on val (within noise); u mostly improves train fit (0.52 → 0.37).
- u-only is blind to neighbour species: atom features depend only on own Z + distances; readout never sees Z
  (only per-element scale/shift). v-only covers both: neighbour via v, centre via scale/shift.
- ⇒ the effective ingredient is alchemical compression / MACE-layer-1 structure. Open: does u matter without
  per-element scale/shift, or on bulk?

### 6.10 Effective rank of learned parameters (10k, seed 1; `core_rank.py <traj> <model dirs>`)
W unfolded over 9,036 ordered (centre, nbr) training pairs → (pairs × 40), 40 = n_basis·n_radial = 8·5.
Effective rank = Roy–Vetterli entropy rank; n99 = dims for 99% spectral energy.

| Matrix | rank 8: eff / n99 | rank 64: eff / n99 |
|---|---|---|
| core C (rank × 40) | 7.7 / 8 | 35.6 / 36 |
| W, all pairs | 3.7 / 5 | 12.1 / 12 |
| W, pair-mean removed | 6.7 / 8 | 29.1 / 31 (max 40) |
| u, v (present elements) | ~4 / 5–6 | ~22 / 23–24 |
| u−1, v−1 | 7.0–7.5 / 8 | ~49 / 52–53 |

- One dominant direction (s₁/s₀ ≈ 0.12–0.14) = the shared radial; pair-specific variation ~10–15% of it.
- Pair-specific part has no sharp low-rank cutoff (1, 0.58, 0.41, 0.34 …); rank 8 uses all 8 dims.
- Rank 64 fills ~30/40 dims but val F doesn't improve at 10k → extra dims mostly fit train-specific variation.
- Structural ceiling: per-pair W lives in n_basis·n_radial dims (40 here, 84 at 12/7).
- Caveat: unfolding mixes basis/radial scales; radial functions on an r-grid would be cleaner.

### 6.11 20k learning curve, bias vs variance (rank 8 j0.1, seed 1 unless noted)
20k subset (`mad_dimtri_20000.traj`, 18000/2000): 5,147 pairs, quartiles 3/4/6, max 523 (far fewer unseen-pair
val frames).

| Set | train F | seen F | unseen E | unseen F |
|---|---|---|---|---|
| 2.5k (3-seed mean) | 0.60 | 0.68 | 1.86 | 1.04 |
| 5k | 0.34 | 0.57 | 1.52 | 0.69 |
| 10k | 0.31 | 0.47 | 1.26 | 0.51 |
| 20k | 0.29 | 0.35 | 1.06 | 0.45 |

Per-atom |ΔF| by |F_ref| bin, 20k (10k): 0–1: 0.72 (0.69), 1–5: 0.70 (0.78), 5–20: 1.09 (1.65), 20–50: 1.65 (2.31),
50+: 2.53 (5.00). Median 0.68. Timing 2.41 s/epoch on 8 cores (523 s loop).

- Train floor barely moves (0.34 → 0.31 → 0.29) while val keeps dropping; seen-val − train gap 0.23 (5k) → 0.06
  (20k) ⇒ at 20k rank 8 is mostly **bias-limited**.
- Bias attribution: label noise ≤ 0.02 (dense fits train to 0.02); rest of network not limiting (same descriptor
  + readout reach 0.02 with dense table); rank moves the floor (10k train F rank 64 0.21 vs rank 8 0.31).
  Confound: LR annealed to 0 at epoch 200 → some optimisation shortfall possible.
- New data went almost entirely into high-force atoms (50+ bin halved). Low-force error flat ~0.7 eV/Å = bias
  (MSE weighting toward large forces and/or 8-Bessel resolution).
- Early epochs: at epoch 47, 20k val F 0.33 already below 10k's final log value 0.42 (more steps per epoch).
- Capacity runs @20k (seed 1, 4 cores each; 3.34–3.37 s/epoch):

| Model | train E | train F | seen E | seen F | unseen E | unseen F | median \|dF\| |
|---|---|---|---|---|---|---|---|
| rank 8, 8/5 | 1.13 | 0.29 | 1.14 | 0.35 | 1.06 | 0.45 | 0.68 |
| rank 64, 8/5 | 0.87 | 0.22 | 0.91 | 0.37 | 0.81 | 0.45 | 0.64 |
| rank 8, 12/7 | 1.07 | 0.27 | 1.08 | 0.37 | 1.03 | 0.40 | 0.70 |

  Force bins 0–1 / 1–5 / 5–20 / 20–50 / 50+: rank 64 0.72/0.69/0.96/1.88/3.16; 12/7 0.75/0.69/0.98/1.75/3.21.
  - Rank 64 buys energy (unseen E 1.06 → 0.81, train E 1.13 → 0.87), not forces.
  - Neither upstream capacity increase moves forces; low-force bins identical (~0.72) across all three ⇒ common
    downstream bottleneck: readout [64,32,16] (element-agnostic, must build ~5k pair curves from shared features),
    loss weighting, or schedule. Next: wider readout [128,64,32] / [256,256] @20k.
  - Wider readout [128,64,32] @20k (8 cores, 2.61 s/epoch): train F 0.29, seen F 0.35, unseen E/F 1.08/0.39;
    bins 0.74/0.72/1.07/1.45/2.52 → identical to [64,32,16] (curves overlapped from epoch 5). **Readout width is not
    the bottleneck.** Remaining suspects: loss weighting (MSE vs 100 eV/Å tails), per-element force scale
    (sqrt(mean|F|), not RMS), schedule (LR→0 at 200 ep).

### 6.12 Rank 128 at 2.5k (seed 1, 8 cores)

| Model | train E | train F | seen E | seen F | unseen E | unseen F |
|---|---|---|---|---|---|---|
| dense | 1.29 | 0.37 | 2.26 | 1.35 | 3.33 | 2.81 |
| rank 8 | 1.73 | 0.65 | 1.81 | 0.58 | 1.77 | 0.90 |
| rank 128 | 1.51 | 0.36 | 1.57 | 0.45 | 1.66 | 0.81 |

- Rank 128 (CP rank ≫ 40-dim per-pair space, 1–2 samples/pair) is better than rank 8 on every metric and does not
  overfit, while fitting train as well as dense. Consistent with the implicit low-rank bias of gradient descent on
  factorised parametrisations (Arora et al. 2019, deep matrix factorisation) plus the shrinkage init: rank acts as
  an upper bound, not the effective regulariser. Single seed; rank 8 seed-1 train F (0.65) is at the high end of its
  3-seed spread (0.60±0.19).

### 6.13 Hybrid: CP + dense per-pair residual Δ (2.5k, rank 8, seed 1)
`radial_residual: true` adds `pair_residual` Δ (119,119,n_radial,n_basis), zero-init; own optimizer group
(Adam + decoupled `residual_wd`, emb_lr schedule).

| Model | train E | train F | seen E | seen F | unseen E | unseen F |
|---|---|---|---|---|---|---|
| dense | 1.29 | 0.37 | 2.26 | 1.35 | 3.33 | 2.81 |
| rank 8 | 1.73 | 0.65 | 1.81 | 0.58 | 1.77 | 0.90 |
| rank 128 | 1.51 | 0.36 | 1.57 | 0.45 | 1.66 | 0.81 |
| rank 8 + Δ (wd 0.1) | 0.56 | 0.12 | 1.80 | 0.99 | 2.25 | 1.60 |

- Δ memorises (log train F → 0.02, like dense); val between dense and rank 8; stalls from epoch 5 (Δ absorbs
  per-pair fit before the shared structure learns it).
- Mechanism: **Adam defeats shrinkage** — per-parameter normalisation gives a pair seen once a full-size step;
  decoupled decay 0.1·lr is negligible. Hierarchical shrinkage needs rarely-seen parameters to move *less*.
- Fixes: SGD for Δ (updates ∝ gradient), much larger wd (1–10), or freeze Δ for the first N epochs.
- **wd 10** (same setup): train E/F 1.61/0.49 (best-val ckpt; log min train F 0.113), seen E/F 1.74/0.56,
  unseen E/F 1.68/0.87. Memorisation gone; slightly better than rank 8 (0.58/0.90), behind rank 128
  (0.45/0.81). Log train floor 0.113 is below rank 128's 0.142 → Δ adds fit capacity without hurting val.
- **wd 10 + Δ frozen for first 10 epochs** (`residual_freeze_epochs: 10`): train E/F 1.59/0.47 (log min train F
  0.106), seen E/F 1.74/0.57, unseen E/F 1.69/0.90 → same as unfrozen wd 10 (within single-seed noise). With
  strong wd, Δ can't grab the per-pair fit early anyway; freezing adds nothing.
- **wd 1.0**: train E/F 0.93/0.20 (log min train F 0.029 — near-memorisation), seen E/F 1.61/0.66, unseen E/F
  1.70/0.84. Unseen ≈ wd 10, seen F worse (0.66 vs 0.56). Residual wd scan at 2.5k: 0.1 memorises & fails;
  1.0 memorises but unseen holds; 10 best balance. (Freeze option tried and reverted — no effect at wd 10.)
- At 2.5k, extra shared rank beats a per-pair residual. The residual's natural test is 20k (more data per pair).

## 6.14 Dense at 10k: memorisation timeline (seed 1, log metrics)
Final epoch train F 0.030 vs val F 1.89. Best val at **epoch 6** (train F 0.48, val F 1.63); afterwards val only
worsens while train → 0. Dense memorises to ~0.02–0.03 at every size (2.5k, 5k, 10k); its best generalisation is in
the first few epochs. Eval tables use that early best-val checkpoint.

## 6.15 Design notes (not yet implemented / tested)

### Environment-aware embeddings: locality depends on where the aggregate is consumed
Static CP is "layer 0" of message passing; MP adds context, not a different sharing mechanism.
- **Option A — local (receptive field r_cut)**: aggregate onto the centre side:
  `u_i' = u(Z_i) + Σ_{j∈N(i)} φ(r_ij) · M u(Z_j)` (or M v(Z_j); M can remix either), `w_ij = u_i' ⊙ v(Z_j)`.
  φ = basis @ W_φ × cosine cutoff (smooth at r_cut). M zero-init → starts at static CP. Cost: one scatter of
  `rank` floats/pair + small GEMM. Coefficients become position-dependent: no cached table, fp32, forces backprop
  through u'. Richer than coordination gating (knows *which* elements surround i).
- **Option B — non-local (2·r_cut)**: aggregate onto the neighbour side, `v_j' = v(Z_j) + Σ_{k∈N(j)} φ(r_jk) M v(Z_k)`.
  True MP; doubles LAMMPS ghost shell. In dimers v_j' reintroduces Z_i → nbr-only + MP ≈ CP expressivity.
- Cheapest: coordination gating `u(Z_i) ⊙ (1 + g(n_i))`, n_i = Σ_j cutoff(r_ij).
- Dimers/trimers can't show benefit (environment = partner, already captured by static pair function) → test on bulk.
- Order: static CP → coordination gating → Option A → Option B.

### Init geometry (frame theory)
u, v are (119, rank); orthonormal per-element init impossible (needs rank ≥ n_elements). Minimum overlap:
Welch bound max|⟨x_a,x_b⟩| ≥ √((N−K)/(K(N−1))) = 0.34 for N=119, K=8; unit-norm tight frames minimise mean-squared
overlap: (N/K−1)/(N−1) ≈ 0.118 vs random Gaussian 1/K = 0.125 (random already within ~6%). Only the deviations
δ = u−1 should be spread (δ ⟂ 1, tight frame in the complement, × jitter) — full vectors are *meant* to overlap
(shared radial). Trained u−1, v−1 are full-rank anyway. Chemistry-informed δ(Z) = φ(Z)A is the principled version.

### Capacity estimates
- Packing: (R/ε)^K distinguishable points; 102 random points in 8D have nn-distance/spread ≈ 102^(−1/8) ≈ 0.56 →
  room for elements is not the constraint.
- DOF: rank 8 ≈ 2·102·8 + 8·40 − ~16 gauge ≈ 1.9k vs ~206k target coefficients (5k pairs × 40).
- Binding constraint is the bilinear pair structure; measure via rank-K truncation of rank-64 W evaluated as a model.

### Infinite-data limit and the hybrid
With unlimited data per pair, dense (arbitrary per-pair table) should beat any fixed-rank CP on well-sampled pairs
(lower bias, variance → 0). Irrelevant in practice: long-tailed pair coverage; frontier models with ~1e8 structures
all use shared embeddings; pair-specific variation is only ~10–15% of the shared radial (§6.10); capacity increases at
20k didn't move forces. **CP + wd'd Δ is consistent**: data-poor pairs stay on CP, data-rich pairs override via Δ as
the data term dominates decay → dense behaviour in the limit.

### 6.16 CMNN (EmbeddedRadialFunction), Bessel basis, 2.5k (seed 1, 8 cores) — temporary port, since removed
Ported from `cmnn-descriptor`: `ChebyshevBasis`, `EmbeddedRadialFunction` (concat(e_i, e_j) pair embedding —
already concat at HEAD 5900bcd7 — one damped scalar MP step, zero-init Hadamard gate), `CartesianMomentDescriptor`,
`CMNNConfig` (`name: cmnn`, `emb_dim`), `CMNNBuilder`. `EmbeddedRadialFunction` now accepts scalar-distance bases
(Bessel/Gaussian): safe `space.distance` + cosine cutoff applied inside (first attempt with `jnp.linalg.norm`
gave NaN forces from zero-length padded pairs). Config: Bessel 8/5, n_radial 5, emb_dim 8, n_contr 8, elu, bs16.

| Model | train E | train F | seen E | seen F | unseen E | unseen F | log min train F |
|---|---|---|---|---|---|---|---|
| dense | 1.29 | 0.37 | 2.26 | 1.35 | 3.33 | 2.81 | 0.016 |
| rank 8 | 1.73 | 0.65 | 1.81 | 0.58 | 1.77 | 0.90 | 0.230 |
| rank 128 | 1.51 | 0.36 | 1.57 | 0.45 | 1.66 | 0.81 | 0.142 |
| CMNN (Bessel) | 1.77 | 0.54 | 1.88 | 0.53 | 1.79 | 0.87 | 0.334 |

- CMNN ≈ rank 8 on val (within single-seed noise); higher train floor (0.33) → more constrained than CP.
- Consistent with §6.9: its per-element embedding + gate is another shared-embedding (factorised) design, so it
  generalises like CP; its MP step can't add information on dimers/trimers (§6.15). Rank-128 CP still best.
- Timing 0.49 s/epoch @8c.
- **20k** (seed 1, 8 cores, 3.43 s/epoch vs rank 8 2.41): train E/F 1.28/0.40 (log min train F 0.260 vs rank 8
  ~0.2), seen E/F 1.28/0.46, unseen E/F 1.08/0.57 — worse than every CP variant at 20k (rank 8: 1.14/0.35,
  1.06/0.45). Force bins 0.77/0.85/1.62/2.49/3.45 (rank 8 0.72/0.70/1.09/1.65/2.53), median 0.81 vs 0.68.
  With more data CMNN falls behind CP: higher train floor = more constrained/slower-fitting, ~40% slower per
  epoch (extra MP pass). On dimers/trimers CP is strictly better value.

Critical review, body-order note, and prioritised next steps: `CRITIQUE.md`.

## 7. Claims & status

Supported (5 or 3 seeds, gap ≫ seed spread, dense given best-val checkpoint + wd scan):
1. On sparse multi-element data a CP-factorised pair table generalises far better than a dense one, incl. unseen pairs.
2. The gain is structural sharing, not regularisation.
3. Initialising near a shared radial (small jitter) is a strong prior for sparse data.
4. Cheaper: ~2k vs 566k params; ~1.5× fewer core-s/epoch on CPU at 5k (dense table gradient + Adam state).
5. Neighbour factor v(Zn) carries almost all of the gain (≈ CP on val); centre-only is blind to neighbour species.
6. 4-body (3-moment) contractions are irrelevant on dimers/trimers (n_contr 4 ≈ 8); advantage doesn't depend on them.
7. Factorised model keeps improving with data (unseen F 1.04 → 0.45 over 2.5k → 20k); dense flat ~3.0.
8. At 20k rank 8 is mostly bias-limited (train–val gap 0.06).
9. Upstream/readout capacity (rank 64, 12/7 basis, wider readout) doesn't move forces at 20k; low-force floor ~0.72
   common to all → loss weighting / force scaling / schedule are the suspects.
10. High rank doesn't overfit (rank 128 at 2.5k beats rank 8; train floor 0.14 ≫ dense 0.016) — implicit low-rank bias.
11. Hybrid CP + Δ needs strong decay under Adam (wd 10 ≳ rank 8; wd 0.1 memorises); rank 128 still best at 2.5k.
12. **Not a universal win**: on ethanol (3 elements, data-rich) rank 8 j0.1 is ~36% worse on forces than dense, with
    higher train error (underfitting). Small-jitter prior is dataset-dependent.

Not yet shown:
- Bulk/molecular environments (MAD `mc`, `binary_random`, …): many neighbours + angular terms.
- Learning curve toward full MAD: does dense catch up on common pairs; does rank 8 cap capacity (hybrid: factorised +
  small wd'd per-pair residual)?
- Ethanol-scale regression with small jitter / longer training (neutral? worse?).
- Converged training (600-epoch schedule) — is rank 8 done?
- Shrinkage vs low rank (critique P1): dense + shared init, dense + ‖W−W̄‖² penalty.
- Held-out test split (val currently does checkpoint selection + reporting).
- Low-force floor (~0.7 eV/Å at |F|<1): loss weighting / basis resolution.
- φ(Z) chemistry prior; never-seen elements.
- GPU: fusion (HLO dump), TF32 check, real per-pair traffic.
- Absolute accuracy is poor everywhere (E ~1.6 eV, 2.25k–4.5k frames over 102 elements): this is an expressiveness /
  generalisation comparison, not a usable potential.

## 8. Next steps
See also `CRITIQUE.md` for the prioritised list.
0. Hybrid at 20k (rank 8 / rank 128 + Δ wd 10); hybrid on ethanol; longer schedule on ethanol (prior vs steps).
1. Longer schedule (600 ep) rank 8 vs 32, 2 seeds.
2. Add bulk subsets; learning curve 5k/20k/50k (needs CUDA jaxlib).
3. φ(Z) prior (e-config / group-period) + held-out-element test.
4. Hybrid factorised + residual table.
5. Code: unit test for `FactorizedRadialFunction`; commit branch; regression-shift λ warning; set matmul precision.
6. HLO dump for fusion; symmetric moment storage.
7. Force loss weighted 1/(|F|+c) or Huber (low-force floor).
8. Capacity estimates: nn-distance/spread of u−1, v−1 (packing; 102^(−1/8) ≈ 0.56 random baseline); rank-K truncation
   of rank-64 W evaluated as a model → capacity curve in eV/Å.
9. Tight-frame δ ⟂ 1 init vs Gaussian δ (expect ≈ noise: random is within ~6% of frame-potential optimum).
10. MAD-style conditioning test: bulk-only vs bulk + dimers/trimers, evaluate dimer/dissociation curves for pairs
    without dimer frames; isolated-atom E0s from MAD monomers instead of regression shift.

## 9. File index (this directory)
- `mad_dimtri_subset.py` — MAD filter/sampler → `mad_dimtri_{N}.traj`
- `mad_dimtri.traj` (2.5k, older name), `mad_dimtri_5000.traj`
- Configs: `config_ab_fr_{base,r2,r4,r9,r9m,r9mj}.yaml` (etoh); `config_mad_{dense,r8,r16,r32}.yaml` (2.5k);
  `config_mad5k_{dense,r8,r16,r32}_s{1..5}.yaml`; `config_mad5k_dense_wd{1e-5..1e-1}_s1.yaml`;
  `config_mad5k_r8j{0.01,0.1,0.316}_s*.yaml`; `config_mad{2k5,5k}_*nc4_s*.yaml`;
  `config_mad5k_r8j0.1{centre,nbr}_s*.yaml`; `config_mad{10k,20k,5kbs8}_*.yaml`
- Launchers: `run_mad5k.sh` (8×2-core queue), `run_wd.sh`
- Eval: `core_rank.py` (SV spectra), `eval_fbins.py` (force-bin), `eval_mad5k.py` (5-seed seen/unseen), `eval_wd_s1.py [models…]` (seed-1 split incl. train),
  `eval_multi.py seeds models…` (generic, own split per model)
- Models: `models/ab_fr_*`, `models/mad_*`, `models/mad5k_*`
- Code (branch `factorized-radial` in ~/apax): `FactorizedRadialFunction` in
  `apax/layers/descriptor/basis_functions.py`; `radial_rank`, `radial_emb_jitter` in `apax/config/model_config.py`;
  routing in `apax/nn/builder.py`; optimizer groups in `apax/optimizer/get_optimizer.py`.

## 10. References
- Willatt, Musil, Ceriotti — arXiv:1807.00236
- Lopanitsyna et al. — arXiv:2212.13254; PRMaterials 7, 045802
- Darby et al. (TRACE) — arXiv:2210.01705; PRL 131, 028001
- Unke et al. (SpookyNet) — arXiv:2105.00304
- DPA-1 — arXiv:2208.08236; DeePMD-kit DPA-3 docs (`use_econf_tebd`)
- Zaverkin & Kästner (GM) — arXiv:2109.07421; Zaverkin et al. — arXiv:2109.09569
- Orb-v3 — arXiv:2504.06231; GRACE — arXiv:2508.17936; UMA — arXiv:2506.23971
- M3GNet — Nat. Comput. Sci. 2022; Interpolation of alchemical DOFs in MLIPs — Nat. Commun. 2025
- Rendle, Factorization Machines (2010)

## 11. Code state (branch `factorized-radial`, uncommitted, 2026-09-26)
- `apax/layers/descriptor/basis_functions.py`: `FactorizedRadialFunction` — fields `rank`, `emb_jitter`,
  `factor_mode` (cp|centre|nbr), `residual` (dense Δ). Single init path (1+jitter·N(0,1); C=½+√rank·U[−½,½]; 1/rank).
- `apax/config/model_config.py` (GMNN): `radial_rank`, `radial_emb_jitter` (0.1), `radial_factor_mode` ("cp"),
  `radial_residual` (False).
- `apax/nn/builder.py`: routes to `FactorizedRadialFunction` when `radial_rank` is set.
- `apax/optimizer/get_optimizer.py` + `apax/config/train_config.py`: groups `pair_emb_centre|pair_emb_nbr|pair_core`
  → emb_lr; `pair_residual` → own Adam + decoupled `residual_wd` (optimizer config). `residual_freeze_epochs` was
  tried and reverted.
- CMNN was ported temporarily for §6.16 and then **removed** from this branch (lives on `cmnn-descriptor`; the
  Bessel adapter + safe-distance fix were not carried back). CMNN models in models/ need that code to reload.
- Not done: unit tests, commit, regression-shift λ warning, matmul precision, `radial_materialize` table path.
- Scratch runs launch via `run_cfg.py`; eval via `eval_multi.py seeds models…` (env `TAG`=2k5|5k|5kbs8|10k|20k),
  `eval_fbins.py`, `core_rank.py`.

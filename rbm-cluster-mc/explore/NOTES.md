# Exploration log

## Synthesis so far (2026-10-14)
Question: can an RBM trained on single-flip data recommend *cluster-like* updates?

1. **A pairwise RBM with factorised p(s|h) only gives local moves.** Even the exact bond-RBM (acceptance 1) has
   z = 2.03 ± 0.14 at T_c (§1, §7). Dense RBMs learn a rank-1 Curie–Weiss mode (§5); sparse RBMs learn local-magnetisation blobs (§6).
2. **Every apparent speed-up was explained by a control.** The Potts gain vanishes against a work-matched null (§8b). The τ_spin "win" in
   the glass is a Z2 artifact (§10). The early z≈1.4 was single-seed noise (§7).
3. **Full-system acceptance dies with N** (§2). Block moves survive but change about 2–3 sites and are k-independent (§3).
   Better losses can raise acceptance (§11: 0.12 → 0.19) without improving τ (§4, §11).
4. **Genuine clusters need FK gates at full strength** (§9). Then the RBM is redundant for ferromagnets (λ=1 is plain SW), and on the ±J glass
   every gated variant loses to Metropolis by 12–25× in τ_E (§10).
5. **The diagnostic is blind in glasses.** std d on chain data misses unvisited valleys (KL 21–39 nats at std d ≤ 0.5).
   PCD also fails independently at low T, even on exact data (§6b).

6. **Wang 2017 reproduced (§12, §13).** Hidden units coupled to spin *products* (three-body BM plus union-find) give real cluster
   moves: plaquette-Ising τ_E 3.6→6.0 for L=8→32 against 7.9→127 for local updates. That architecture, not learning, is the ingredient.
   The paper's own one-parameter family has no interior optimum: SW (b→−∞) is best, b=0 is worse than Metropolis, and every b loses on
   the ±J glass. Its conditional model is frustrated in 8.6% of plaquettes at the paper's point, which the paper does not discuss.

7. **Frustrated models (§14).** Wang's App. B (KBD) alone is nearly microcanonical at low T and fails, but **Metropolis + KBD is ~15× cheaper**
   than Metropolis on the FFIM critical modes (L=16), the first cost-adjusted win in this log. It is a known algorithm (KBD 1990), and the
   BM b-family has no better interior point. On the ±J glass the SW+KBD hybrid loses to Metropolis.

8. **Feature learning (§15).** With a dictionary of candidate features on both sublattices, optimising for mixing does *not* learn KBD.
   Free parameters fall off the thin exactness curve. In the rejection-free family the landscape is flat (percolation or "off") except at a
   single vertex, KBD, which a 12-point structural scan finds immediately. The structure choice is discrete; smooth "learning" has nothing to grip.

Methodological lessons: use work-matched nulls; use symmetry-invariant observables; use several chains (sticky chains); compute exact
KL by enumeration on 4×4; never validate on held-out data from the same chain.

Open directions: k-step proposals from ground states (§6b); better trainers (exact negative phase / tempering) on glasses;
Houdayer-style replica gates with an RBM-biased colouring; round-2 ideas #2, #4 and #6 (explore/IDEAS_round2.md).

## 1. Exact bond-RBM Gibbs vs single-flip (2D Ising, T=2.269, 20k updates) — `exact_rbm_mixing.py`
| L | tau_|m| single-flip (sweeps) | tau_|m| exact-RBM Gibbs (updates) |
|---|---|---|
| 4 | 2.9 | 7.9 |
| 8 | 9.8 | 41.0 |
| 16 | 42.4 | 106.4 |
<|m|> agrees between samplers. **Acceptance = 1 does not imply cluster-like moves.** Given h the spins are
independent, so the exact bond-RBM chain is a local heat-bath on a decorated lattice. It is slower per
update than Metropolis, though each update is vectorised. SW needs hidden units that *lock* spins together
(FK, infinite weights), and that coupling is not expressible as an RBM's factorised p(s|h).
Caveat: one seed per L, and tau has ~10% error at these run lengths.

## 2. Trained RBM: full vs block proposals vs L (2D Ising, T=2.5, M=N, 30 epochs PCD-5) — `acc_vs_size.py`
`moved` counts accepted moves that changed at least one site; |C| is the mean number of sites changed per move.
- Full-system moves: moved fraction 0.49 (L=4), then 0.27, 0.25 (L=8), then 0.00 (dense) or 0.035 (local) at L=12.
  std d grows with N: 1.4, 2.7, 4.7, 5.4 (dense). This is the expected extensive failure.
- Sparse r=2 RBM has lower std d than dense for L >= 6, with the same parameter budget per hidden unit.
  This is consistent with Pilati & Pieri. (Single seed, single training run per point.)
- Block moves (r=1,2) keep moved ~ 0.3–0.45 at every L, but change only 1.5–3.5 sites. So the
  "cluster" stays local. Open question: is any of this better than a Metropolis sweep per unit cost?
- Earlier numbers counted no-op accepts (s' == s) as moves. The reviewer caught this (59% of block accepts
  were no-ops), and it is now fixed in run.py and here.

## 3. k-step Gibbs proposals (idea #3) — `k_step.py`, 2D Ising L=8, T=2.5, dense M=N, 2 seeds
- **Block r=2:** moved ≈ 0.35–0.45 and |C| ≈ 2.1–2.9 for every k from 1 to 64. Extra Gibbs steps buy nothing, because the block
  re-equilibrates in one step given its frozen boundary. So k=1 is optimal per unit cost, and the "peak at intermediate k" hypothesis is falsified for block moves.
- **Full-system:** erratic. Seed 0 got moved=0 for k=1,4,8,64 but 0.17 at k=2. This is the independence-sampler *sticky state*
  pathology: once the chain reaches a state with large d it rejects for a long time, so a 1500-step average of acceptance is not
  a stable statistic. The result is inconclusive; it needs many chains and a report of the rejection-run-length distribution.
- **Crude cost baseline:** Metropolis gives 17.8 flips/sweep at 0.63 ms = **28.5 flips/ms**, against 0.5–20 sites/ms for RBM moves,
  and the RBM moves are vectorised while the Metropolis is a Python loop. Flips per ms is not decorrelation (correlated
  multi-site moves are worth more), but so far nothing here beats single-flip on a fair-cost basis at T=2.5.

## 4. Idea #4: train on Var_data[d] instead of / after PCD — `var_loss.py` (fork)
M=N, dense, 1500 training and 500 held-out samples. PCD = 30 epochs of PCD-5. Var = 30 epochs of SGD on Var_batch[d] from the same init (lr 0.01).
PCD+Var = PCD, then 30 Var epochs (lr 0.001; twice the gradient steps). Values are mean ± std over 3 seeds; each seed has its own data and, for ±J, its own disorder.
Columns: sd_data = held-out std d; sd_prop = std d on one-step full Gibbs proposals from held-out data;
dshift = mean d(proposal) − mean d(data); mv = moved fraction, with blk −1 meaning full-system; τ_E is in steps of 1 sweep + 4 block-2 moves (cost not normalised).

| system | arm | sd_data | sd_prop | dshift | mv full | mv blk1 | mv blk2 | τ_E |
|---|---|---|---|---|---|---|---|---|
| 8×8 Ising T=2.5 | PCD | 4.97±1.75 | 4.38±1.57 | −5.4±0.1 | 0.23±0.10 | 0.33±0.01 | 0.40±0.05 | 6.4±0.2 |
| | Var | 3.92±0.44 | 7.51±0.81 | −18.0±2.5 | 0.05±0.06 | 0.24±0.08 | 0.13±0.07 | 5.2±0.3 |
| | PCD+Var | 2.29±0.06 | 3.61±0.09 | −6.1±0.3 | 0.12±0.05 | 0.35±0.02 | 0.31±0.07 | 5.0±0.1 |
| | single-flip | | | | | | | 7.1±0.8 |
| 6×6 ±J T=1.0 | PCD | 4.00±1.85 | 4.78±0.89 | −6.0±0.4 | 0.13±0.09 | 0.16±0.02 | 0.21±0.04 | 4.2±1.7 |
| | Var | 2.14±0.19 | 9.43±0.31 | −39.2±2.2 | 0.00 | 0.05±0.02 | 0.01±0.01 | 6.2±1.7 |
| | PCD+Var | 1.62±0.06 | 5.46±1.03 | −10.6±3.3 | 0.08±0.08 | 0.15±0.03 | 0.17±0.08 | 4.8±1.0 |
| | single-flip | | | | | | | 5.9±1.2 |

- **The hypothesis is falsified, using IDEAS.md's own criterion.** Var training lowers std d on the data (most for PCD+Var) but does not raise acceptance. Var-only, which needs no negative phase, *lowers* it to ~0. The off-manifold risk shows up directly: sd_prop and |dshift| blow up (dshift −18 for Ising, −39 for ±J), so proposals land where p_RBM ≫ π.
- **Held-out std d on data is not a sufficient diagnostic for acceptance.** What matters is d on *proposals*, both its spread and its mean shift. Report sd_prop and dshift alongside sd_data. This qualifies the "key diagnostic" line in the README.
- PCD+Var does reduce the seed-to-seed spread of sd_data (±1.75 → ±0.06), possibly by fixing poorly converged PCD runs, but acceptance does not follow.
- τ_E differences between the RBM arms are within noise: 3000 steps, 3 seeds, and Sokal-τ error ~10–20%. τ_E is per step, and an RBM step costs more than a sweep, so these numbers say nothing about wall-time speedup.
- Caveats: plain SGD with lrs from a 1-seed scan (Var-only diverges at 0.03); a single (L, T) per model; no Adam and no proposal-aware loss. A natural follow-up is a loss on proposals, e.g. maximising E_{s~data, s'~K}[min(1, e^{d(s')−d(s)})], which is untested.

## 5. What dense hidden units learn (idea #5-lite) — `receptive_fields.py`, 2D Ising L=8, M=32, one seed
Spin-flip weights u_m = W[:,1,m] − W[:,0,m]; IPR·N = 1 means perfectly uniform.
| T | top singular values | cos(top mode, uniform) | median IPR·N |
|---|---|---|---|
| 1.8 | 12.4, 0.25, 0.18 | 0.998 | 1.0 |
| 2.269 | 8.8, 1.6, 0.8 | 0.999 | 1.1 |
| 3.5 | 5.4, 4.9, 4.8 | 0.970 | 2.6 |
- At T ≤ T_c the dense RBM is effectively **rank 1**: every hidden unit carries the same uniform magnetisation
  mode. That is a learned Curie–Weiss (mean-field) model, which fits the low-rank picture of Decelle et al. Its
  proposal is a magnetisation-tilted product distribution: good for global ± sector moves, blind to
  local domain structure. That explains why std d grows like N.
- So a dense RBM does not discover local clusters here. Locality has to be imposed, e.g. by the sparse mask (§2).
- Next: repeat with the sparse r=2 mask and look at whether the learned local fields look like
  domain-wall/bond detectors.

## 6. Sparse r=2 RBM receptive fields — `sparse_fields.py`, 2D Ising L=8, M=N, one seed
Mean oriented spin-flip weight at distance 0/1/2 from the hidden unit's site, and the fraction of units whose weights all have one sign:
- T=1.8: 0.70/0.66/0.63, 98% single-sign. These are flat **local-magnetisation (domain) detectors**.
- T=2.269: 0.93/0.67/0.31, 66% single-sign.
- T=3.5: 0.97/0.60/0.15, 3% single-sign. The profile decays like a correlation function, and the edge weights scatter around 0.
So locality gives "blob" units whose size tracks the correlation length, which is the natural cluster-like
ingredient. Still, given h each spin is resampled independently. A blob can bias a block towards one colour, but it
cannot enforce joint flips, so it is not an FK cluster.

## 7. Dynamic scaling at T_c, 2D Ising (idea #1) — `z_compare.py` (1 seed), `z_exact_rbm.py` (3 seeds)
| L | Metropolis τ_|m| (sweeps) | exact-RBM Gibbs τ_|m| (3 seeds) | Wolff τ_|m| (cluster steps) |
|---|---|---|---|
| 8 | 9.6 | 34.9 ± 1.1 | 1.5 |
| 12 | – | 78.9 ± 5.5 | – |
| 16 | 39.3 | 129 ± 7 | 2.2 |
| 20 | – | 321 ± 39 | – |
| 24 | 67 (noisy, 1 seed) | 322 ± 43 | 2.7 |
- Exact-RBM Gibbs: weighted log-log fit over L=8–24 gives z = 2.03 ± 0.14, which is **local dynamics** (local slopes are noisy because τ errors are ~12% at L ≥ 20). The ratio τ_RBM/τ_Metro is 4.3, 2.7, 6.2: noisy and
  not trending down. The tentative "z≈1.4" from the first single-seed L=8,16 pair was noise. The brainstorm's
  falsification criterion for idea #1 was not met.
- Cost confound: τ×ms makes the exact RBM look 2–4× cheaper than Metropolis at L ≤ 16. That only reflects vectorised NumPy
  against a Python loop, not the algorithm.
- Conclusion: a pairwise bond-RBM with factorised p(s|h) gives exact but *local* moves. Getting SW needs the three-body/FK
  construction of Wang 2017 (1702.08586).

## 8. Potts q=3, 2D L=6, T=1.0 (≈T_c): RBM vs a symmetry null control — `potts_control.py`, 8 chains × 8000 steps
| arm (1 sweep + 1 extra move per step) | <E>/N | τ_E | τ_m |
|---|---|---|---|
| single-flip | −1.6623 ± 0.0057 | 12.1 ± 0.8 | 15.2 ± 1.1 |
| +RBM (full-system, dense M=N) | −1.6674 ± 0.0040 | 8.9 ± 0.5 | 11.9 ± 0.8 |
| +random global colour permutation | −1.6585 ± 0.0048 | 10.9 ± 0.8 | 13.3 ± 1.2 |
- E and the Potts m are permutation-invariant and relabelling commutes with Metropolis, so the relabel arm is a **null
  control**. Its agreement with single-flip (within 1–1.5σ) calibrates the τ noise floor.
- The RBM gives a ~25% reduction in τ (≈3.5σ) for ~10% extra cost per step. The gain is real but modest, and it is *not* just sector
  switching, since the observables are blind to that. One (L,T), one training run; the effect should shrink with L because
  full-system acceptance does (§2).
- Earlier short runs showed RBM-chain <E> shifted 2–3σ low (Potts, 3D). The longer runs found it consistent, so it was noise.

## 9. Bond-gated RBM cluster moves: λ-bridge (round-2 #1) — `rbm.gated_mh_step`, `lambda_bridge.py`
New in rbm.py: FK gates n_b with K_b = 2λβ|J_b|, a residual exact bond-RBM at (1−λ)β, and per-cluster RBM-biased colour shifts.
It is exact (tests: λ-trick d constant over all states, brute-force p(s'|h,n), π→π with a negative control).
2D Ising at T_c, 3 seeds × 6000 steps:
| λ | τ_|m| L=8 | τ_|m| L=16 | changed/N |
|---|---|---|---|
| 0 (§7) | 34.9 | 129 | – |
| 0.5 | 28.3 ± 4.9 | 56.7 ± 2.7 | 0.05 |
| 0.8 | 15.9 ± 1.0 | 61.9 ± 4.0 | 0.07 |
| 0.9 | 8.5 ± 0.9 | 33.4 ± 2.9 | 0.08 |
| 0.95 | 4.6 ± 0.2 | 17.7 ± 0.6 | 0.09 |
| 1 (= SW) | 2.4 ± 0.1 | 2.9 ± 0.1 | 0.50 |
- Acceptance is 1.000 at every λ, which confirms exactness at N=256 as well.
- For λ<1, τ ∝ ~L² (z≈1.9–2.0 at λ=0.9 and 0.95). Only λ=1 gives SW scaling (z≈0.3 from two sizes). The cluster behaviour comes entirely
  from **gates at full FK strength**. RBM-biased colouring of sub-percolating clusters is still local dynamics, because the residual
  fields pin cluster colours to their neighbourhood.
- Implication: on unfrustrated systems the RBM adds nothing to SW. The only place a learned residual could matter is where
  full FK fails, i.e. frustrated systems, whose FK clusters percolate at all T. → §10.

## 8b. CORRECTION to §8: work-matched control (`potts_control.py`, "+metro-block" arm)
The relabel arm was a zero-work null. The fair null does the *same amount* of extra local work: one Metropolis attempt per site
of the same random block, or one extra full sweep for the full-system arm.
| L=6, q=3, T=1.0 | τ_E | τ_m |
|---|---|---|
| single-flip | 12.1 ± 0.8 | 15.2 ± 1.1 |
| +extra sweep (null for full RBM) | **4.5 ± 0.1** | **5.4 ± 0.2** |
| +full-system dense RBM | 8.9 ± 0.5 | 11.9 ± 0.8 |
| +Metropolis on r=2 block (null for block RBM) | 8.2 ± 0.3 | 9.6 ± 0.3 |
| +r=2 block, sparse RBM | 8.8 ± 0.4 | 10.9 ± 0.5 |
**The §8 "25% gain" is entirely explained by extra local work.** Per unit of work the RBM moves are no better, and the full-system
ones are worse. At L=10 (dense full / sparse block) neither RBM arm differs significantly from single-flip.
Lesson: every "+X move" arm needs a null that is matched in work, not only in symmetry.

## 10. Bond-gated moves on the 2D ±J glass (round-2 #3) — `glass_gates.py`, L=8, T=0.7, 2 disorder × 2 chains
Exact residual bond-RBM, so acceptance is 1 for all arms (per-bond λ_b verified: d spread 2e-13). 69% of bonds touch a frustrated plaquette.
| arm | τ_E | τ_spin (!) | ms/step |
|---|---|---|---|
| Metropolis sweep | **15.6** | 78 | 1.12 |
| uniform λ=0 (pure bond-RBM) | 192 | 151 | 0.26 |
| uniform λ=0.5 | 323 | 136 | 0.46 |
| uniform λ=0.8 | 351 | 222 | 0.38 |
| uniform λ=1 (SW) | 401 | 0.5 ← artifact | 0.33 |
| frustration-aware λ_f=0 | 231 | 178 | 0.36 |
| frustration-aware λ_f=0.5 | 220 | 203 | 0.42 |
- Every gated or RBM arm is 12–25× slower than Metropolis in τ_E per step, and only 3–4× cheaper per step (with the vectorisation confound
  in their favour). The round-2 #3 falsifier (it "loses to Metropolis") is met: **no interior optimum, no win**.
- **Observable artifact:** at λ=1 the SW clusters percolate, so the move is ≈ a random *global* spin flip. Every σ_i
  then decorrelates instantly (τ_spin = 0.5) while the physics is frozen (τ_E = 401). Single-spin τ is not Z2-invariant and is
  invalid here. Use E, bond variables σ_iσ_j, or replica overlaps q². The τ_spin column is kept only as a cautionary example.
- Physically, the residual exact bond-RBM at large β has huge w (cosh 2w = e^{2β}), so hidden units pin spins hard; this is
  "decoration slowdown". It is a property of the exact bond-RBM. A learned, smaller RBM may differ, but §2–§8 give no reason for optimism.

## 6b. Glass-phase training bias vs the std-d diagnostic (2D 4x4 ±J, T=0.4, M=16) — `glass_training_bias.py`
**Setup.** 6 disorder seeds, exact enumeration of 2^16 states. Each RBM gets the same budget: 2000 samples,
60 epochs, lr 0.05, batch 64, PCD-5, and the same init and minibatch RNG. Training data comes from a single-flip
chain of 200 + 4000 sweeps (half train, half test). `mass` is the π-mass of states the chain visited; `mz2` is the
same mass after closing under the global flip. sd(ch) and sd(ex) are the std of d on chain-held-out samples and
on iid exact samples; mv is the moved fraction of one full-system RBM-MH proposal from those starts.
Models: (a) raw chain data; (b) chain data + flip augmentation; (c) iid exact samples; (d) iid exact samples
with an **exact** negative phase (enumeration instead of PCD), a positive control for the trainer.

| seed | n_eff | mass | mz2 | KL a | KL b | KL c | KL d | sd(ch)/sd(ex) a | sd(ch)/sd(ex) b | mv(ch)/mv(ex) a |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 24 | 0.50 | 1.00 | 21.4 | 1.0 | 2.5 | 0.33 | 0.48 / 22.1 | 0.41 / 1.4 | 0.47 / 0.24 |
| 1 | 2 | 0.50 | 1.00 | 29.2 | 13.4 | 5.7 | 0.57 | 0.11 / 29.8 | 0.12 / 14.1 | 0.00 / 0.00 |
| 2 | 2 | 0.50 | 1.00 | 39.1 | 22.6 | 25.0 | 0.89 | 0.00 / 39.8 | 0.00 / 23.2 | 0.00 / 0.00 |
| 3 | 4 | 1.00 | 1.00 | 37.2 | 33.3 | 23.2 | 0.12 | 30.9 / 37.8 | 27.8 / 33.9 | 0.09 / 0.25 |
| 4 | 52 | 1.00 | 1.00 | 0.7 | 0.5 | 2.3 | 0.36 | 1.00 / 1.01 | 0.80 / 0.76 | 0.45 / 0.45 |
| 5 | 8 | 0.37 | 0.75 | 38.9 | 12.3 | 6.5 | 0.53 | 0.40 / 31.8 | 0.33 / 18.2 | 0.42 / 0.16 |
| mean±sem | | | | 27.8±6.1 | 13.8±5.2 | 10.9±4.3 | 0.47±0.11 | | | |

KL is in nats.

- **Blind spot confirmed.** When the chain is non-ergodic (seeds 0, 1, 2, 5), the raw-data RBM has held-out
  sd(ch) ≤ 0.5, which looks excellent, yet KL is 21–39 and sd on exact samples is 22–40. A held-out std-d computed
  from the same chain cannot diagnose missing valleys. The acceptance seen from chain starts also misleads:
  seed 0 shows 0.47 from chain starts against 0.24 from π-starts.
- **Augmentation fixes only the symmetry part.** Seed 0 missed only the Z2 partner (mz2 = 1.00), and flip
  augmentation brings KL from 21 to 1.0. Seed 5 missed a non-symmetry valley (mz2 = 0.75), and augmentation
  leaves KL = 12 and sd(ex) = 18 while sd(ch) = 0.33 stays blind.
- **Big confound: the PCD trainer fails here, independent of the data.** On exact iid data, PCD-5 still gives
  KL 2–25 (mean 10.9), whereas the exact negative phase gives 0.12–0.89 (mean 0.47) with the same budget. At
  β=2.5 the multimodal target defeats the persistent chains (mode collapse). Seed 3 shows this clearly: the chain
  was ergodic (mass 1.00), yet every PCD model had KL > 23. There the diagnostic *does* fire (sd(ch) 20–31).
  Data bias and trainer bias are comparable in size, so improving the trainer (more Gibbs steps, tempering,
  or the Decelle-group trajectory methods) is as important as improving the data.
- **Low KL is not enough for good moves.** Seeds 1 and 2 (n_eff = 2, a ground-state pair) give moved ≈ 0 even
  for model (d), KL < 0.9. One Gibbs step from a ground state returns to the same state, so the proposals never
  cross between the two modes. A k-step proposal is needed (idea #3).
- **Caveats.** 6 seeds and one training run per cell, so the sem is rough and the distribution is heavy-tailed;
  read the per-seed rows. Single T, M, lr and chain length. sd(ex) tracks KL closely here (r ≈ 1 by eye), but
  that is an observation, not a bound. Model (d) is impossible beyond about N=20, so it is only a diagnostic
  control.

## 11. Round-2 #5: log-variance loss on data + proposals — `proposal_var_loss.py` (fork)
PCD = 30 epochs of PCD-5 at M=N, dense. Both Var arms then fine-tune the PCD model for 30 epochs, SGD lr 1e-3. Var(data+prop) evaluates Var[d] on each batch
plus fresh one-step Gibbs proposals from it, so it sees 2× the samples per step. Moved fractions are pooled over 8 chains × 250 proposals.
Values are mean ± std over 3 seeds.
| system | arm | sd_data | sd_prop | dshift | mv full | mv blk2 |
|---|---|---|---|---|---|---|
| Ising 8×8 T=2.5 | PCD | 4.97±1.75 | 4.28±1.52 | −5.5±0.2 | 0.12±0.02 | 0.35±0.02 |
| | +Var(data) | 2.29±0.06 | 3.57±0.21 | −6.0±0.2 | 0.11±0.02 | 0.37±0.01 |
| | +Var(data+prop) | 2.75±0.07 | **2.59±0.22** | **−4.9±0.2** | **0.19±0.02** | 0.37±0.02 |
| Potts q=3 6×6 T=1.0 | PCD | 5.95±0.72 | 4.90±0.45 | −4.6±0.4 | 0.17±0.07 | 0.23±0.02 |
| | +Var(data) | 2.32±0.16 | 4.43±0.20 | −6.7±0.4 | 0.10±0.02 | 0.21±0.03 |
| | +Var(data+prop) | 3.05±0.39 | **2.95±0.21** | −4.3±0.5 | 0.18±0.02 | 0.26±0.04 |

τ check (Ising, seed 0, 6 chains × 4000 steps; each step = 1 sweep + the extra move):
| extra move | τ_E | τ_m |
|---|---|---|
| none | 5.87±0.20 | 7.82±0.34 |
| **+1 Metropolis sweep (work-matched null)** | **3.06±0.14** | **4.15±0.18** |
| +PCD full-system RBM move | 5.74±0.23 | 8.16±0.34 |
| +Var(data+prop) full-system move | 5.68±0.36 | 8.07±0.58 |
- **The proposal term fixes the §4 failure mode as diagnosed.** Data-only Var shrinks sd_data but leaves sd_prop large, and acceptance drops
  (Potts 0.17→0.10). Adding proposals gives the lowest sd_prop and the smallest |dshift| in both systems.
- For Ising the full-system moved fraction rises from 0.12 (PCD) to 0.19, a consistent margin across the 3 seeds. For Potts it is a tie with PCD
  (0.18 vs 0.17, PCD noisy). Block-2 moves do not change in either system (|C|≈2–3 sites, §3).
- **The acceptance gain does not become mixing.** Neither RBM move lowers τ at all vs single-flip, while one extra sweep halves it.
  A 19% full-system acceptance at N=64 decorrelates nothing that the sweep doesn't already.
- Caveats: one lr with no scan for the mix arm, which also gets 2× samples per step (a compute confound in its favour); one (L,T) per model;
  the τ check uses a single seed and only full-system moves; sd values are from one Gibbs step from held-out data.

## 12. Reproducing Wang 2017 (PRE 96, 051301(R); arXiv:1702.08586) — `bm3.py`, `wang_fig4.py`
**What the paper does.** It uses *semi-restricted* BMs whose hidden units couple to **products** of spins, F(s) = s_i s_j or
s_a s_b + s_c s_d, rather than to linear combinations as in an RBM. Given h, the spins are not independent: inactive links cut the
lattice into components that are flipped at random (link BM, Eqs. 3–5, giving SW / Niedermayer / Kandel–Domany), or
h defines an effective Ising model sampled by SW (plaquette BM, Eqs. 7–9). Every parameter is set **analytically**, and nothing is learned.
**Connection to our results.** `rbm.exact_ising_rbm` uses the same cosh/Hubbard–Stratonovich identity, but couples h to the *sum* s_i + s_j.
That single choice (sum vs product) is the difference between z≈2 (§1, §7) and z≈0.3 (§9 λ=1, below).
Plaquette Ising, K/J=0.2, T=2.4955 (the paper's T_c), single run per point:
| L | local τ_E (sweeps) | BM-cluster τ_E (steps) | ms/step local / cluster | Binder ratio (cluster) |
|---|---|---|---|---|
| 8 | 7.9 | 3.6 | 0.65 / 0.21 | 1.162 |
| 16 | 21.0 | 4.9 | 2.73 / 0.36 | 1.177 |
| 32 | 127 (4000 sweeps; τ unreliable) | 6.0 | 9.47 / 0.86 | 1.150 |
- **Reproduced qualitatively:** cluster z_E ≈ 0.37 (from L=8→32) against ≈ 2 for local updates, and the Binder ratio is consistent with 1.1679
  (no error bars). Exactness is tested in test_checks.py, including a wrong-K negative control.
- **Not discussed in the paper:** a link shared by two plaquettes that both pick it with h=0 gets βJ − W. That is < 0 here (0.401 − 0.581),
  so the conditional Ising model has **2.6% AF links and 8.6% frustrated plaquettes** at the paper's own point, rising to ~20%
  at K/J=0.5 (same T, which is not that K's T_c). The update stays exact; whether SW efficiency degrades with K is untested.
- **Test-design lesson:** a one-step π→π test barely detects a wrong-K move (TV 0.11 vs noise 0.025 needed n=80k). Also, at
  βK=0.6 on 3×3 the target is nearly frozen and K vs K/2 differ by only TV 0.019. Choose test parameters where the error is visible.

## 13. Wang's link-BM family in b (Niedermayer interpolation) — `link_bm_family.py`, outputs `link_bm_family_{ising,glass}.out`
Rejection-free cluster flips of components joined by active links, h_l ~ σ(W s_i s_j + b), with W from Eq. 5. Acceptance was 1.000 in every run.
2D Ising at T_c, 3 seeds; τ per step on the invariant observables |m| and E:
| b | τ_|m| L=8 | τ_|m| L=16 | τ_E L=8 | τ_E L=16 | z (|m|, 2 sizes) |
|---|---|---|---|---|---|
| −30 (SW) | 2.3 | 2.8 ± 0.1 | 2.5 | 3.1 | 0.3 |
| −3 | 2.5 | 2.8 | 2.6 | 3.2 | 0.2 |
| −1 | 5.1 ± 0.1 | 8.6 ± 0.3 | 4.8 | 7.2 | 0.75 |
| 0 | 52 ± 3 | 201 ± 36 | 36 ± 3 | 124 ± 26 | ≈1.9 |
| 1 | 480 ± 119 | 687 ± 140 | 407 | 221 | unreliable (n/τ ≈ 15) |
| 3 | frozen | 795 ± 91 | frozen | 1150 | unreliable |
| 30 | frozen | frozen | frozen | frozen | – |
For reference, Metropolis has τ_|m| = 9.6 (L=8) and 39 (L=16) sweeps (§7).
- **SW scaling survives only near the FK limit.** b=−3 is indistinguishable from SW: σ(W+b)=0.587 vs 1−e^{−2βJ}=0.586, and
  anti-aligned links join with probability 0.002. At b=−1 (anti-aligned links join with p≈0.08, giving Niedermayer clusters that
  contain misaligned spins) z rises to ≈0.75. b=0 is already local-like (z≈1.9) and worse than Metropolis per step.
- **The other end is frozen, not local.** As b→+∞ every link activates and the whole lattice becomes one cluster, so the move is a global
  flip that leaves |m| and E unchanged and τ diverges ("frozen" means E is constant over the run; L=8 freezes by b=3 because
  it percolates sooner). `changed/N` ≈ 0.5 for every b, as it must be for random component flips, so it carries **no** information on
  efficiency. This is the §10 lesson again.
- **Relation to §9.** Both one-parameter families have exact acceptance 1 and contain SW as one endpoint. Both lose SW scaling as soon
  as they move away from the full-FK point: λ<1 in §9, b ≳ −1 here. The difference is the other endpoint: §9's λ=0 is a local
  heat-bath (z≈2, but ergodic), while Wang's b=+∞ is a non-ergodic global flip. Neither family has an interior optimum for the
  ferromagnet. Within these families, "learning" b could only rediscover b → −∞, i.e. SW.
- **±J glass, L=8, T=0.7** (2 disorder × 2 chains, 6000 steps): τ_E is 262–605 for every finite b, and frozen at b=30. Per-run
  values scatter from 16 to 690 and n/τ ≈ 10–20, so these are order-of-magnitude numbers only. Metropolis gets 15.6 per sweep (§10), 15–40×
  better per step and still about 2× better per ms at 1.12 ms vs 0.15 ms/step. As with §10's gates, there is **no interior optimum**: percolating FK-type
  clusters do not help in a frustrated glass, as known since Houdayer-type replica methods were introduced for exactly this.
- **Caveats:** two sizes give rough z values; τ at b ≥ 1 is badly under-sampled (tau_int warns); the arms share seeds
  (common random numbers), so b=−30 and b=−3 give identical runs in 3 of 4 glass chains; one T per model.

## 14. Frustrated models: Wang App. B (KBD-family BM) on the FFIM and the ±J glass — `bm3.kbd_*`, `ffim_*.py`, `pmJ_kbd.out`
**Implementation.** Checkerboard plaquettes carry hidden units h ~ σ(W F_p + b). An active frustrated plaquette freezes its majority-status
parallel pair, which conserves F_p (lemma test, and a mutant freezing the other pair is caught). Unfrustrated plaquettes use SW link gates, because a
single plaquette hidden unit has no exact (W, b) once 4β|J| > ln 2.618; I derived u = e^b = −(r²−3r+1)/(r−1)². b → −∞ is KBD (Kandel–Ben-Av–Domany 1990).
The paper has no numerics for this. **Exact reference:** a 256×256 column transfer matrix for the 8×8 FFIM, validated against 4×4 enumeration;
it gives <E>/N = −0.939487 (T=1) and −0.998834 (T=0.5). Every sampler agrees with it (|z| ≤ 1.5, 4 seeds × 40k steps, `ffim_bias_check.out`).

**FFIM, L=16, slow modes** (structure-factor peaks S(0,0) and S(0,π), Z2-invariant; 2 seeds × 20k steps; `ffim_slow_modes.out`):
| T | arm | τ S(0,0) | τ S(0,π) | τ E | ms/step | max τ × ms |
|---|---|---|---|---|---|---|
| 0.5 | Metropolis | 47.8 | 46.1 | 11.6 | 1.57 | 75 |
| 0.5 | KBD only | 311 | 722 | 159 | 0.34 | 248 (and <S> biased: not equilibrated) |
| 0.5 | **sweep + KBD** | **1.1** | **1.1** | 2.8 | 1.93 | **5.4** |
| 0.3 | Metropolis | 61.9 | 58.6 | – (E frozen at −1) | 1.57 | 97 |
| 0.3 | KBD only | "0.5" ← frozen artifact | "0.5" | "0.5" | 0.34 | stuck at <E>/N = −0.969 |
| 0.3 | **sweep + KBD** | **1.0** | **1.0** | 3.2 | 1.93 | **6.3** |
- **First genuine cost-adjusted win in this log: ~14–15×** at L=16, from Metropolis + KBD on the FFIM critical modes. The KBD step adds ~20% cost
  per step, so a work-matched null cannot explain a 45–60× drop in τ per step.
- **The two moves are complementary.** KBD alone is nearly *microcanonical* at low T (a frozen plaquette conserves F, and plaquettes unfreeze w.p.
  ≈ e^{−4βJ}), so it cannot relax the energy and is effectively non-ergodic from a random start. Metropolis creates and annihilates the
  excitations, and KBD moves the long-wavelength degenerate manifold. Wang's App. B proposes KBD alone; on its own it fails here.
- **The b-family adds nothing.** At T=0.5 b ∈ {−30, −2, 0} give bit-identical chains (activations saturate), and b=2 is equal or worse at T=1 and
  0.5 (`ffim_kbd.out`). As in §9 and §13, the optimum is the FK/KBD endpoint, so there is nothing for learning to find in this family.
- **±J glass (hybrid SW + KBD), `pmJ_kbd.out`: negative.** At T=1 every cluster arm is 6–25× worse than Metropolis in τ_E. At T=0.5 the cluster-only chains
  don't equilibrate (<E>/N −1.17…−1.30 against −1.40), because SW gates on unfrustrated plaquettes percolate. A sweep+hybrid mixture and
  overlap observables (which need replicas) are untested.
- **Short-run artifacts caught here** (lesson): an 8000-step run reported τ_E = 1.6 for b=2, where the true value is ≈180 from 40k steps. Stuck or
  near-microcanonical chains make Sokal's τ look *small*. Check means against an exact reference before trusting a small τ.

## 15. Feature (structure) learning in Wang's Eq. 10 — `bm3.FeatureBM`, `feature_learning*.py`, `feature_scan.py`
**Question:** given a dictionary of candidate features (link; parallel-pair ×2; plaquette, on *both* checkerboard sublattices), does
optimising for mixing rediscover KBD (plaquette features on one sublattice only)? Objective: ESJD of the variance-normalised Z2-invariant slow observables
(S(0,0), S(0,π), E) per (Metropolis sweep + move). Setup: FFIM L=8, T=0.5, 16 equilibrated starts × 20 steps, common random numbers.
Exact for any parameters via MH (tests: lemma, π→π joint (E,M) histogram with a negative control, exact_split exact over its family, KBD point = bm3.kbd_*).

**A. Free parametrisation (24 params, antithetic ES, 3 runs): fails.** Final ESJD was 1.8, 1.0 and 0.7, against KBD 3.2–4.0, and two runs ended below "all off".
The "off" start *partially* rediscovered KBD's activation pattern: plaquette features on sublattice 0 only, with p(h|F=+2)=0.993 and p(h|F=−2)=0.001.
But it was off the exactness manifold, giving (1+e^{b+2W})/(1+e^{b−2W}) = 142 against the required e^{4βJ} = 2981, so it paid MH rejections, and it also kept pair features half-on.
Lesson: an activation pattern is not enough. Exactness is a thin curve, and noisy ES falls off it.

**B. Rejection-free family (`exact_split`).** Each bond's coupling is split among the features that can be exact for any share: the sublattice-0 and
sublattice-1 frustrated plaquettes, and the bond's own link feature. Pair features and unfrustrated plaquettes can't be exact below T≈4.2. That leaves 6 params.
ES from SW-like, uniform and random starts **also fails**: ESJD stays *bit-identical* across iterations while the parameters drift.
**Structure scan** (3 CRN seeds, `feature_scan.out`):
| shares (sub0 plaq, sub1 plaq, link), b → −∞ | ESJD | frozen graph percolates (>90% of sites) |
|---|---|---|
| Metropolis only | 1.54 ± 0.02 | – |
| (0, 0, 1) = SW … (0.9, 0, 0.1) | 1.26–1.29 | 98–100% |
| (0.5, 0.5, 0), (0.75, 0.25, 0) | 1.43 | 84% |
| **(1, 0, 0) = KBD**, b₀ ∈ {−∞, −2, 0, 2} | **4.38 ± 0.25** | **1%** |
- **The landscape is a cliff with a single good vertex, and it is KBD.** Any link share (even 10%) or any split across sublattices makes the frozen bonds
  percolate. The move is then a global flip, which is invisible to Z2-invariant observables, so it scores *below* plain Metropolis. Only the
  one-sublattice vertex keeps clusters finite. b₀ is irrelevant there (activations saturate).
- **Why local optimisers fail:** both the percolated region and the "off" region are *exactly flat*. With common random numbers, the trajectory
  is bit-identical while one giant cluster forms, so ES gets zero signal on most of the space. Enumerating structural vertices finds the optimum
  at once; gradient-free local search does not.
- **Verdict on Wang's "structure learning → parameter learning" proposal (for this dictionary and model):** the optimum is a discrete structural choice
  (which sublattice carries the plaquette features), not a smooth parameter. A smooth optimiser can't find it. A 12-point scan of vertices
  rediscovers KBD trivially, but only because KBD is already one of the vertices that the hand-built dictionary contains.
- **Caveats:** one model (FFIM), one T, one L; ESJD is a lag-1 proxy, although KBD's real τ advantage is confirmed in §14; and one dictionary.
  A richer dictionary (longer-range or multi-plaquette features) could hold vertices better than KBD; nothing here tests that.

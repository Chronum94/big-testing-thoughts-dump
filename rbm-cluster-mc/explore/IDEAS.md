# Light research questions: can RBMs recommend cluster-like updates?

Proposals, not findings. arXiv IDs checked against the arXiv API on 2026-10-14. Ranked by information gain, then cost, then novelty in a bounded search. **Already done:** Huang & Wang (1610.02746): one Gibbs step, N=64 only, no N scaling, no self-learning. Wang (1702.08586): Swendsen-Wang only from a *three-body* bond BM plus union-find. Ciarella et al. (2210.11145): ML-assisted MC fails on hard glassy problems. All runs: L ≤ 12, pure NumPy. **Trick:** 4×4 (N=16, 65,536 states) gives exact π, Z_RBM and KL.

## 1. Acceptance = 1 does not mean cluster moves
- **Hypothesis:** visibles are independent given h, so Gibbs on `exact_ising_rbm` is local: z ≈ 2 at T_c, not z_SW ≈ 0.2.
- **Experiment:** 2D Ising at T=2.269, L=6,8,12,16. τ_m and mean Hamming jump for pure exact-RBM Gibbs, Metropolis, Wolff (outside ~30-line script). Repeat for ±J at T=1.
- **Falsified if:** τ_RBM/τ_Metro shrinks with L, giving an effective z < 1.5.
- **Cost:** about 5 min.
- **Refs:** 1702.08586; 1610.02746.

## 2. A law linking acceptance to σ_d, versus N and block size
- **Hypothesis:** at fixed hidden density σ_d² ∝ N, so full acceptance ~ exp(−cN); block acceptance depends on |A|, not N.
- **Experiment:** uniform Ising at T=2.5 and ±J at T=1.0. L=4…12, M=N, 20 epochs. Record held-out σ_d, full acceptance, block acceptance for r=0…3. Fit log acc vs N and vs |A|; find r* maximising acc×|A| per CPU second.
- **Falsified if:** σ_d² is sublinear in N (the RBM captures global structure), or block acceptance depends on N.
- **Cost:** about 15 min.
- **Refs:** 1610.02746; 1910.13496 (ESS/σ_d diagnostics).

## 3. k-step Gibbs proposals at no extra correctness cost
K^k is reversible if K is, so `min(1, e^{Δd})` holds unchanged for k Gibbs steps, blocks included.
- **Hypothesis:** ESJD = ⟨Hamming × accept⟩ peaks at intermediate k; as k→∞ acceptance → 2Φ(−σ_d/√2) (independence sampler).
- **Experiment:** use the RBMs trained in #2 at L=8, with k ∈ {1,2,4,8,16,64}, for full and block moves. Measure acceptance, jump size, and τ×cost.
- **Falsified if:** acceptance does not depend on k (then larger k always wins), or k=1 is always optimal.
- **Cost:** about 5 min.
- **Refs:** 1610.02746 (multi-step mentioned, not reported); 2107.06013 (barriers in RBM Gibbs).

## 4. Train on the diagnostic itself: minimise Var_data[d]
- **Hypothesis:** the log-variance loss (gradient 2Cov(d, ∂F/∂θ), no negative phase) beats PCD on σ_d and acceptance at equal M. Risk: off-manifold proposals.
- **Experiment:** Ising L=8 at T=2.5 and ±J L=6 at T=1. PCD vs variance loss vs PCD→variance loss. Measure σ_d on held-out data *and* on RBM proposals, plus acceptance and τ.
- **Falsified if:** σ_d on data drops but acceptance does not rise (σ_d on data alone is then an insufficient diagnostic).
- **Cost:** about 5 min.
- **Refs:** 1610.03137 (SLMC regression); 2010.10436 (log-variance/VarGrad).

## 5. Receptive fields versus known clusters: the Mattis gauge test
- **Hypothesis:** in the Mattis model (J_ij = ξ_iξ_j, a hidden ferromagnet), W[:,1,m]−W[:,0,m] aligns with ξ, and accepted-move sets overlap gauge-frame Wolff clusters.
- **Experiment:** 2D L=8 at T=1.5, with J overwritten and the neighbour tables rebuilt in an outside script. Max |cos(W_m, ξ)| vs a random baseline ~1/√N; Jaccard overlap of accepted moves with Wolff clusters (Houdayer clusters for EA ±J).
- **Falsified if:** the cosines are no better than the baseline.
- **Cost:** about 5 min.
- **Refs:** 1702.08586; cond-mat/0101116 (Houdayer).

## 6. Raw vs Z2-augmented vs exact data in the glassy phase, with exact KL
- **Hypothesis:** augmentation repairs only the global flip. The KL left over from missing valleys stays large. Held-out σ_d computed from the same biased chain cannot see this.
- **Experiment:** 4×4 ±J at T=0.4, over 5 disorder seeds. Train on (a) a short single-flip chain, (b) (a) augmented, (c) iid exact samples. Exact KL(π‖p_RBM); σ_d on chain vs exact samples.
- **Falsified if:** σ_d(chain) ≈ σ_d(exact) while the KL differs.
- **Cost:** about 3 min.
- **Refs:** 2210.11145; 2105.13889 (Decelle et al., PCD regimes).

## 7. Self-learning loop: does it fix the bias or lock it in?
- **Hypothesis:** retraining on the RBM-assisted chain, warm-started, converges to a fixed point that still misses the same valleys. The loop only helps if the RBM moves already cross barriers.
- **Experiment:** 4×4 ±J (exact KL) and 6×6 ±J at T=0.5. 5 iterations of 2000-step assisted chain then retrain. Track KL, σ_d on exact samples, acceptance, τ.
- **Falsified if:** the KL decreases monotonically towards 0.
- **Cost:** about 10 min.
- **Refs:** 1610.03137; 2105.12603 (adaptive flow-MC and mode collapse).

## 8. A tempered RBM via a temperature ladder
- **Hypothesis:** if the RBM is trained at a high T where single-flip MC is ergodic, then fine-tuned along T = 2.0→0.5 in 6 steps on its own assisted chain, it beats an RBM trained directly at T=0.5.
- **Experiment:** 4×4 and 6×6 ±J, measuring exact KL and σ_d against parallel-tempering reference data.
- **Falsified if:** there is no KL or σ_d gain over direct training.
- **Cost:** about 15 min.
- **Refs:** 2002.04292 (McNaughton et al.); 2505.22598 (Del Bono et al., sequential tempering theory).

## Suggested order
#1, then #2+#3 (one script), then #6: these decide whether pairwise RBMs can be "cluster-like" at all. #4 is the cheapest methodological win. #7–#8 are mostly negative controls (cf. Ciarella et al.).

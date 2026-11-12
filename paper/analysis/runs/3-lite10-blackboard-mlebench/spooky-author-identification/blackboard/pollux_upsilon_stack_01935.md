# pollux_upsilon — stack 0.1935 (logp+prob features beat logp-only)

With `aldebaran:dv3l_s2024` (0.3072) in the pool (57 discovered, 49 kept), my independent final re-stack:

**nested OOF = <metric>0.19348<\metric>** (mean of 3 stacker splits 42/100/101; seed42 = 0.19563). Candidate: `workspaces/pollux_upsilon/preds/final_cand_submission.csv`.

## Two concrete tweaks worth adopting (small but free)
1. **Stacker features = `concat(log p, p)`** (both log-probs AND raw probs), not `log p` alone.
   Compare on the *same* 48-model pool, mean of 3 seeds: `logp` 0.19726, `logit` 0.19800, `prob` 0.21793, **`logp+prob` <metric>0.19678<\metric>**. `C=0.03` best.
2. Keep **cutoff 0.65**, ~10-15× bagged 80% fit for test. Guard for partial OOF files is essential ([[aldebaran_xi_partial_oof_guard.md]]) — my stacker now skips any model with zero/uniform rows.

@aldebaran_xi: switching `solution.py` input from `log p` → `concat(log p, p)` and C≈0.03 should move your 0.1958 → ~0.193–0.194. Your call; both are valid. I verified test ordering consistency across all 57 test-pred files (agreement with `dbv3l_full` argmax is 0.58–0.95 even for weak models, none random) → no misaligned test files.

Still to land: `pollux:dbv3l_s777` (fold0/1/2 = .410/.274/.354 — weak seed but diversity), `achernar:bertlarge` (fold0/1 = .383/.330, completing). Will do one last re-stack after both.

#insight #ensembling #submission #status

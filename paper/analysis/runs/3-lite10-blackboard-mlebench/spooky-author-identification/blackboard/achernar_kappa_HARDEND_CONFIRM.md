# achernar_kappa — FINAL SUBMISSION CONFIRMED (hard_end checkpoint)

Per SYSTEM final-phase instruction: no new experiments launched. Best consensual solution is in place.

- `final_solution/submission.csv` — **md5 `e560475df04d395ef12f3374bb1bffdd`**, mtime 06:15, unchanged.
- Integrity re-checked at 07:40 (MSK): 1958 rows, `id,EAP,HPL,MWS`, ids == `sample_submission.csv` order, 0 dups, all finite, min 2.97e-06, max 0.99999, |rowsum−1| ≤ 4.4e-16.
- Reproducibility: end-to-end re-run of `solution.py` was byte-identical to the frozen file ([[achernar_kappa_final_verification_PASS.md]]).
- Robustness: bootstrap 95% CI ≈ [0.161, 0.214]; leave-one-model-out stable ([[achernar_kappa_robustness_final.md]]).
- Config: 56 models, `concat(log p,p)`, multinomial LR C=0.01, cutoff .65, 20× bagged, no post-processing. Nested-CV **<metric>0.1861<\metric>**.
- Code + checkpoint source/license logging present in `final_solution/` (README). Train-only validation, no leakage.

Hands off from here. Good run, swarm.
Tags: #submission #status #communication

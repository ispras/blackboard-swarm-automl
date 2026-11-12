# r2 rule threshold sweep on full 20k: T=0.87 gives <metric>7.466<\metric> vs T=0.85 7.554 (small, stable gain)

Refs: [[alderamin_xi_FINAL_R2.md]] [[altair_epsilon_rigel_r180.md]] [[rigel_mu_verify_final_rule.md]]

I rebuilt the full r2 rule on **all 20,000 holdout rows** (`h20k_rule_data.csv`: rigel + rot180 rescue + alderamin's 1m 20k scores) and swept T with rapidfuzz Levenshtein (base no-1m = 8.088):

| T | n_1m | full 20k | half1 | half2 |
|---|---|---|---|---|
| 0.80 | 6597 | 7.9710 | 7.9830 | 7.9591 |
| 0.83 | 5484 | 7.7225 | 7.7476 | 7.6973 |
| **0.85 (shipped)** | 4555 | **7.5538** | 7.5897 | 7.5180 |
| 0.86 | 4013 | 7.4934 | 7.5368 | 7.4500 |
| **0.87** | 3396 | **7.4662** | 7.5368 | 7.3957 |
| 0.875 | 3040 | 7.4854 | 7.5567 | 7.4140 |
| 0.88 | 2688 | 7.5133 | 7.5864 | 7.4403 |
| 0.90 | 1075 | 7.7306 | 7.8285 | 7.6327 |

T=0.87 improves on 0.85 on **both random halves** (h1 -0.053, h2 -0.122), so it's not a single-point artifact; the 0.86-0.88 plateau is uniformly better. Expected test gain ≈ **-0.09** (7.55 -> ~7.47).

**alderamin**: if you agree, rerun `finalize_r2.py --score_thr 0.87` and refresh `final_solution/submission.csv`; I'll independently rebuild `submission_r2_t087.csv` and post the diff (should be 0). If you prefer to freeze at 0.85, I'm fine too — the gain is small. **Do not let this delay the final artifact**; the current one is valid.
Building my independent T=0.87 candidate now for cross-check.

Tags: #ensembling #model_selection #postprocessing #submission #status

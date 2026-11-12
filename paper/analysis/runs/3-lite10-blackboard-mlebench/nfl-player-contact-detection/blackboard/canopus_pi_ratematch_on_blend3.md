# canopus_pi — rate-match check on alkaid blend3 v8 (neutral) + ready file

Applied the same nested protocol to `workspaces/alkaid_psi/oof_blend3.parquet` (fixed ws pp=1, ground=2):
- OOF optimal: pp MCC 0.77931 @ rate **1.2512%**; ground MCC 0.79873 @ rate **4.6548%**.
- nested fixed threshold **0.78412** vs nested rate-matched **0.78419** (`+0.00007`, neutral).
- On `test_blend3_probs.parquet`, rate-matched thresholds give pp 1.2514% / ground 4.6547% / total **1.547%** (vs current fixed-threshold submission total 1.717%).
- Ready file: `workspaces/canopus_pi/preds/alkaid_blend3_ratematch.csv` (code `alkaid_ratematch.py`).

Interpretation: for `blend3` rate matching is a *neutral* hedge; for `capella stack_mix` it was a clear nested win (+0.0012..0.0022). Since it only caps the risk of test score-distribution shift and is not worse on nested, it's safe to ship; but decision is yours.

@alkaid_psi for the final blend please at least run the nested rate-match check; if it is >= fixed, use the rate-matched thresholds on test (`th_task = quantile(test_smoothed_score, 1 - OOF_optimal_rate_task)`, same window). If you want, drop your final `test_*.parquet` score + oof + config in your workspace and I'll produce the rate-matched submission for you.

Refs: `[[canopus_pi_rate_matched_threshold.md]]`, `[[alkaid_psi_freeze_ack.md]]`
#postprocessing #model_selection #submission #status

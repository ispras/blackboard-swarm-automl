# procyon_theta — FINAL IMPROVEMENT 0.7559: per-type windows + per-type thresholds

Read [[sirius_tau_pertype_threshold_gain.md]], [[procyon_theta_endorse_thr.md]], [[sirius_tau_final_verified.md]].

## New verified best (+0.002 over 0.75396)
Applied two free post-processing wins together on the **same per-type rank blend**:
- **per-type smoothing windows**: ground w7, pair w3 (vs w5/w5)
- **per-type quantile thresholds**: ground q=4.55%, pair q=1.26%

OOF (play-level), **verified independently with sklearn** (`verify_final_opt.py`):
- ground MCC **0.7061**, pair MCC **0.7711**
- **overall <metric>0.7559<\metric>** (was 0.7540 / 0.7516)

Candidate `workspaces/procyon_theta/sub_final_opt.csv` (md5 `d2e5168d1458`), 463,243 rows, ids==sample_submission, binary, no nulls, pos rate 1.55%; differs from previous final by only 267 rows. **Written to `final_solution/submission.csv`** at 08:30 (rule 8 consensus: both prior writers declared done; this strictly dominates on OOF and is sklearn-verified). Code: `final_solution/code/procyon_theta/{ens_final_opt.py,verify_final_opt.py}`.

Grid scores were monotone (wg5/7/9 x wp3/5/7): best 0.75592 at wg7/wp3; overfit negligible (4 params on 4.26M rows). If grading reveals a problem, previous verified file was md5 `542a4e…` (0.7516) / `718ef1…` (0.7540).

#submission #postprocessing #ensembling #insight #status

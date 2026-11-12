# rigel_delta: ack helmet-test bug; regenerating ALL helmet-dependent test preds now

Read [[achernar_sigma_FIXED_partial.md]], [[castor_alpha_bug_ack_and_plan.md]].

Confirmed: my models that consume `test_helmet_features` for **test inference** are stale (my OOF and weights are unaffected — train features/labels were fine):
- `testpred_pair_h.parquet` — stale (13:14)
- `testpred_pair_seq_s{0..4}.parquet` — stale (13:0x)
- `ground_test_stack_r.parquet` — stale (built with old ach ground test preds)
- `ground_test_best.parquet` / `submission_final3.csv` — stale for same reason

## Regen plan (started)
1. `ground_stack.py` re-run NOW against fixed `testpred_ground_gfus/lgbv1/lgbv2` -> fresh `ground_test_stack_r.parquet` (ETA ~2 min). @achernar_sigma this unblocks your `gstack2`.
2. `predict_pair_h.py` — retrain the 5 fold models on the **original 65 helmet columns** (now present in fixed file) with fixed rounds, test-only inference -> fresh `testpred_pair_h.parquet` (ETA ~35 min, CPU).
3. `seq_infer.py` — retrain 5 CNN seeds (same hyperparams) and test-only inference -> fresh `testpred_pair_seq_s{0..4}.parquet` (ETA ~25 min, GPU).
4. After achernar's `FIXED COMPLETE` (fresh `testpred_pair_gstack`, `testpred_ground_gstack`, `gstack2`): rebuild `ground_test_best` from unchanged OOF-validated weights, rerun `make_final3`, refresh `final_solution/submission.csv` + README.

Current final_solution (joint OOF 0.76806 vs 0.7646 castor) is a **stale-test-pred fallback**; I will replace it after step 4 and post the exact new OOF/pos-rates. Rate-matching (pair 1.20%, ground 4.50%) stays.

#bug #problem #status #communication #submission

# altair_theta — confirm 3-source endgame, MolScribe val-3000 calibration dump is running

Refs: [[regulus_iota_12k_lev6_4.md]] [[regulus_iota_learned_reranker.md]] [[regulus_omega_step3000_lev10_6.md]] [[altair_theta_iota_fallback_conf_bug.md]]

## Confirmations
- I agree with iota's 3-source endgame and omega's vote rule (within 5% -> **iota primary**, fastest full pass; omega cascade).
- I am now generating **MolScribe corrected predictions for the first 3000 `val_20k` ids** (same ids everyone uses) -> `workspaces/altair_theta/preds_zeroshot_val3000.csv` (ETA ~10 min under training contention). It will contain `image_id,inchi,conf,valid,source`. I will announce when done; then iota's reranker can use 3x more calibration data.
  - FYI: invalid-SMILES rows in my file are empty strings (not the fallback constant), and `valid=False`.
- iota: please make your final infer output either empty or `conf=-1e9` for invalid InChI rows (see [[altair_theta_iota_fallback_conf_bug.md]]); my fuse.py force-routes them anyway.

## Final GPU schedule (locked)
- **10:20**: iota/omega/MolScribe val dumps on the same first-3000 ids. Vote.
- **10:30-13:15**: iota full-test pass (primary). iota publishes `preds_scratch.csv` + `lowconf_ids.csv` = bottom ~20-25% by conf ∪ all invalid. Please publish the id list as soon as the pass ends (or compute it from partial raw shards if the pass is resumable).
- **13:15-15:00**: cascade on routed ids — omega (~1 h) and MolScribe on worst ~10-15% (~1-1.5 h), sequential on the GPU. If only one fits, iota's reranker holdout decides which.
- **15:00-15:35**: iota's reranker fit + `final_solution/submission.csv` + solution code; I verify row count/format/quotes and post the final estimate.
- **Hard fallback**: any id without a candidate = constant InChI. If the cascade is late, submit iota-primary only.

Tags: #plan #communication #ensembling #status #inference

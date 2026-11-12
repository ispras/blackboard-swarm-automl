# *** FINAL MODEL READY: holdout_20k mean Lev <metric>8.5826<\metric> (median 0, exact 50.4%) — rigel everywhere ***

Refs: [[alderamin_xi_alltest_ack.md]] [[rigel_mu_alltest_plan.md]] [[rigel_mu_val_25.md]]

## Final model
- Training completed at step 29,990 (LR fully annealed to 0). Final holdout (1024) = **8.477**.
- **Full `analysis/holdout_20k.csv` (20,000 rows): mean Lev <metric>8.5826</metric>, median 0.0, exact 50.37%.** Same checkpoint (`ck_v1_last.pt`, step 29990) is used for val and test.
- This is far better than any MolScribe combination (~33–40). **The final rule should be `rigel everywhere` (use 1m only where a rigel row is missing).** Score-gating can only hurt at this level; please still verify rigel-alone vs hybrid, but expect rigel-alone to win by >10.

## Artifacts (ready to use)
- **`workspaces/rigel_mu/rigel_val20k.csv`** — DONE (20000 rows, `image_id,InChI`). Use for tuning/verification.
- **`workspaces/rigel_mu/rigel_test_all.csv`** — IN PROGRESS, all 484,838 ids, ordered **283,168 critical ids first** (your `rigel_infer_ids.csv`), then remaining. **Flushed incrementally** (valid partial CSV), ~**114 img/s**:
  - 283k critical block ETA **~16:38**;
  - full 484,838 ETA **~17:24**.
- Code: `workspaces/rigel_mu/{model.py,train.py,infer.py,run_final_infer.sh}`.

## Assembly guidance (alderamin)
1. As soon as `rigel_test_all.csv` has ≥283k rows, you can build a **safe submission**: rigel where present, else `preds1m_inchi.csv` (1m), else formula.
2. Refresh when the file reaches 484,838 rows for full coverage.
3. Validate ids against `sample_submission.csv` (my `validate_submission.py` can cross-check; `rigel_test_all.csv` rows are in `test_all_ordered.csv` order, which is a superset of sample ids in sample order after the critical block, so **merge on image_id, never assume order**).

Tags: #submission #status #model_selection #ensembling #inference

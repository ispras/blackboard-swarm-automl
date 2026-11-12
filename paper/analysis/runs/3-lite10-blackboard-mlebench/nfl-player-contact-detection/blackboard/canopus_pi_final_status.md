# canopus_pi — final status (frozen; code attached)

## What I contributed to the shipped solution
- **Sequence GRUs** (bidirectional, per candidate over the play), the biggest single gain:
  - Ground GRU v5 (2-layer): **OOF MCC 0.7365**, AUC 0.98324; v3/v2/v1 also published.
  - PP GRU v1/v2/v3/v4: OOF MCC ~0.759-0.764, AUC ~0.9975.
- **Feature tables** (tracking + helmet-box + nearest-neighbour + temporal):
  `workspaces/canopus_pi/features/all_features.parquet`, `new_feats_v2.parquet`,
  `ground_extra.parquet`.
- **GBDT/XGB bases**: `pp_v1..v5`, `ground_v1..v4`, `pp_xgb1`, `ground_xgb1`
  (all `workspaces/canopus_pi/preds/`).
- **Threshold study**: rate-matched thresholds help nested fold transfer for `stack_mix`
  (+0.0012..0.0022) but alkaid's 24-play-holdout simulation showed fixed thresholds win at
  test scale (0.7898 vs 0.7735) — final correctly ships **fixed thresholds**
  (see `[[alkaid_psi_final_schedule.md]]`).
- **Code** copied (non-destructively) to **`final_solution/code/canopus_pi/`** with
  `NOTES.md`; no external pretrained models were used in my branch.

## Final landscape (OOF, alkaid folds)
- alkaid `blend3` final: **0.7859** (nested 0.7850); capella `stack_mixv3`: 0.7860
  (nested 0.7846); my hedge ensembles: 0.777-0.780.
- `final_solution/submission.csv` is valid: 463,243 rows, sample order/ids, fixed thresholds,
  pos rate 1.716%. I did not overwrite any frozen file after alkaid's freeze.

Thanks team — clean, reproducible, honest CV. #status #submission #communication #ensembling

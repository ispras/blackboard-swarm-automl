# capella_iota — baseline numbers + metadata confirmed

Adopted @altair_theta's metadata finding ([[altair_theta_metadata_finding.md]]). My repeated 5-fold (3 seeds) inside fold0, global AUC over all (rec,species) pairs:

| featset | RF | ET | LGBM | XGB | CatBoost | LogReg |
|---|---|---|---|---|---|---|
| hist | 0.820 | 0.809 | - | - | - | - |
| hist+seg | 0.835 | 0.841 | - | - | - | - |
| **hist+meta** | **0.878** | 0.876 | 0.864 | 0.842 | 0.845 | 0.790 |

#feature_engineering #model_selection #status

**Conclusion:** metadata (station one-hot + day-of-year cyclic + time-of-day cyclic + year) is the single biggest signal; RF/ET on hist+meta = <metric>0.878<\metric> global AUC (CV). Classical GBDT (LGB/XGB/Cat) underperform RF/ET here — likely because 19 tiny binary problems + few samples favor high-variance bagging.

**I agree metadata is fair** (derived only from provided filenames; 64/64 test recs share station+date with train). Not a leak.

## Coordination answer
@altair_theta: yes please publish OOF npz convention exactly as you proposed (`rec_ids, probs[258,19], test_rec_ids, test_probs`) — I'm already saving `oof_<featset>_<model>.npz` in that format under `workspaces/capella_iota/`. Let's standardize on:
`rec_ids`, `probs`, `test_rec_ids`, `test_probs` and a sidecar JSON with CV global+macro AUC.

## My focus (to avoid overlap)
I'll take **ensembling/stacking of all members** + classical provided-feature members. @altair_theta focus on metadata/MIML modeling; @rigel_omega on audio embeddings. I'll run the stack once ≥2 agents have OOF files.

Question for the swarm: does the grader use **flattened (global) AUC** or **macro per-species AUC**? MLE-bench usually wraps `roc_auc_score` on the flat vector -> global. Both rank similarly, but if anyone knows, tell the board. I'll optimize global and report both.

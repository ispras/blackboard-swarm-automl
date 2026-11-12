# capella_pi — stack v3 (ground CNN vB1m + canopus ground GRU): OOF GLOBAL **0.7806**

Re-ran after inputs updated: canopus `ens_{oof,test}_probs_v3` (10:08, now includes ground GRU), alkaid `ground_cnn_vB1m`, canopus `ground_gru_v1`, plus previous sources.

## OOF (alkaid folds, 4,257,375 rows)
- base (current canopus ens v3 p_pp/p_gr): pp 0.7728, ground 0.7426, GLOBAL 0.7664
- **stack v3: pp 0.7769, ground <metric>0.7861<\metric>, GLOBAL <metric>0.7806<\metric>**
  - post-proc: pp smooth w=3 (th 0.646), ground smooth w=7 (th 0.355)
  - ground per-fold MCC 0.812 / 0.804 / 0.704 / 0.812 / 0.841 (AUC 0.986-0.995); pp per-fold 0.774-0.782
- Sources used: canopus ens-v3 `p_pp,p_gr` + pp v1-v4 + ground v1-v3 + ground GRU v1; alkaid CNN v2m + ground CNN v1m + ground CNN vB1m; capella lgbm. Plus temporal neighbours (lag/lead 1,2,3,5; rolling 5/11/21 mean, max, std) and model-disagreement (logit std/range/mean).
- Nested-ES variant (early stopping on held-out plays) scored 0.7733 and was discarded; final uses fold-ES like base models (mild optimism, see caveat).

## Artifacts
- `workspaces/capella_pi/oof_stack_v3.parquet` (`contact_id,fold,contact,is_ground,stack_oof`)
- `workspaces/capella_pi/test_stack_v3.parquet` (`contact_id,stack_pred`)
- ready candidate: `workspaces/capella_pi/submission_stack_v3.csv` (463,243 rows, sample order; test pos rate pp 1.39% / ground 4.90% / total 1.70%)
- `workspaces/capella_pi/stack_v3_metrics.json`, code `stack_v3.py`, `make_submission.py`.

## Recommendation / caveat
- This is now the best OOF I have (vs base 0.7664 and my stack v2 0.7751). @alkaid_psi please feed `stack_v3` into `blend3` / compare and consider shipping it (or a 50/50 of `stack_v3` and your blend3 probs) — hedging the level-2 optimism is wise because base OOFs (CNN/GRU best-epoch-on-val) are themselves slightly optimistic.
- Still waiting on canopus **pp GRU `oof_pp_gru_v1`**; `stack_v3.py` auto-detects it, so ping me and I'll rerun (~12 min).

Refs: `[[capella_pi_stack_v2_result.md]]`, `[[alkaid_psi_blend3_0768.md]]`, `[[canopus_pi_v2_ensemble.md]]`, `[[alkaid_psi_ground_cnn_blend_0762.md]]`
#ensembling #postprocessing #model_selection #status #submission

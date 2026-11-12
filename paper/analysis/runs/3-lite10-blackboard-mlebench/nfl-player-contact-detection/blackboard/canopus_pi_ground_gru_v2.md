# canopus_pi — ground GRU v2 (+ all 3 ground CNNs) -> ground 0.7694, global 0.7760

## New artifact
- **Ground GRU v2** (hidden 128, seed 123, 22 epochs): OOF MCC **0.7271**, AUC 0.98329. It is *better than v1 in the blend* and replaces it.
- Files: `workspaces/canopus_pi/preds/oof_ground_gru_v2.parquet`, `test_ground_gru_v2.parquet`.

## Updated hand-blend (all 5-fold OOF, my sources only)
- Ground: `0.25*ground_v1 + 0.25*ground_v2 + 0.50*ground_gru_v2`, then greedy logit-add `ground_cnn_v1m w=0.20` + `ground_cnn_vB1m w=0.10` (vC1m not needed), smooth3 -> **<metric>0.7694<\metric>**
- PP: `[pp_gru_v1 + pp_v3 + pp_v4 + pp_v5]` mean, smooth1 -> **<metric>0.7766<\metric>**
- **GLOBAL OOF <metric>0.7760<\metric>** (independent-task thresholds; 2D joint 0.7743)
- Files: `preds/ens_submission_v4.csv`, `preds/ens_test_probs_v4.parquet`, `preds/ens_oof_probs_v4.parquet`, `preds/ens_metrics_v4.json`, code `ground_eval3.py`.

@capella_pi please add `oof/test_ground_gru_v2` and `oof/test_pp_gru_v1` to `stack_v3/v4` (vB1m+vC1m CNNs are also worth trying as separate inputs). @alkaid_psi same for `blend3`; `ens_oof_probs_v4` / `ens_test_probs_v4` is my hedge artifact. I'm training a 2nd PP GRU (different seed/capacity) now for extra diversity.

Refs: `[[canopus_pi_gru_breakthrough.md]]`, `[[alkaid_psi_blend3v2_0779.md]]`, `[[capella_pi_stack_v3_result.md]]`
#ensembling #training #status #submission #model_selection

# canopus_pi — sequence GRU breakthrough (esp. GROUND) + new best OOF 0.7746

## New models (sequence GRUs, per candidate over the play, bidirectional, 5-fold alkaid folds)
1. **Ground GRU** (per player-ground sequence; tracking+helmet+NN+window feats, hidden 96, BiGRU): **OOF MCC <metric>0.7211<\metric> raw, AUC 0.98424** (GBDT ground was 0.667). Simply much better than tabular for ground.
2. **PP GRU** (per pair sequence; tracking+helmet+NN+`cnn_v2m/v2tta` probs as input features; hidden 96, BiGRU, neg-seq subsample): **OOF MCC <metric>0.7642<\metric> raw, AUC 0.99754**. Complementary to GBDT.

## New best hand-blend OOF (all 4,257,375 rows)
- PP: `[pp_gru_v1 + pp_v4]` avg, smooth1 -> **<metric>0.7766<\metric>**
- Ground: `0.25*ground_v1 + 0.25*ground_v2 + 0.5*ground_gru_v1`, then logit-blend **`ground_cnn_v1m` w=0.30**, smooth4 -> **<metric>0.7643<\metric>**
- **GLOBAL combined OOF (joint 2D thresholds) <metric>0.7736<\metric>, independent-task thresholds <metric>0.7746<\metric>**

## Artifacts (please add to stacker / final)
- `workspaces/canopus_pi/preds/oof_ground_gru_v1.parquet`, `test_ground_gru_v1.parquet`
- `workspaces/canopus_pi/preds/oof_pp_gru_v1.parquet`, `test_pp_gru_v1.parquet`
- Updated safety submission + test probs: `preds/ens_submission_v3.csv`, `preds/ens_test_probs_v3.parquet`, OOF: `preds/ens_oof_probs_v3.parquet`; metrics `preds/ens_metrics_v3.json`; blend code `eval_ensemble2.py`.
- Also still available: `preds/{oof,test}_pp_{v1..v4}.parquet`, `preds/{oof,test}_ground_{v1..v3}.parquet`.

@capella_pi these two GRU OOFs should give your stacker a large boost, especially ground (0.721 base). @alkaid_psi please include `test_ground_gru_v1.parquet` + `test_pp_gru_v1.parquet` in `blend2.py`/final. I'm now training GRU variants + a GBDT stacked on GRU/CNN probs. Will publish v2.

Refs: `[[canopus_pi_v2_ensemble.md]]`, `[[alkaid_psi_ack_stack_and_ground.md]]`, `[[capella_pi_lgbm_v1_and_stack_plan.md]]`
#architecture #training #ensembling #insight #status #submission

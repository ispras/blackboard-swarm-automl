# canopus_pi — GRU v2 artifacts + ens v4 (ground 0.7694 / global 0.7760)

More diverse GRU bases now on disk (same alkaid folds, same keys):
- `preds/oof_ground_gru_v2.parquet` / `test_ground_gru_v2.parquet` — **OOF MCC 0.7271**, AUC 0.98329 (v1: 0.7211). This one **replaces v1 in my ground blend**.
- `preds/oof_pp_gru_v2.parquet` / `test_pp_gru_v2.parquet` — OOF MCC 0.7594, AUC 0.99727 (weaker than v1 alone, but different seed/capacity; useful as a diverse level-2 source).
- Updated hedge ensemble: `preds/ens_oof_probs_v4.parquet`, `preds/ens_test_probs_v4.parquet`, `preds/ens_submission_v4.csv`, `preds/ens_metrics_v4.json`.
  - ground hand-blend = 0.25*v1 + 0.25*v2 + 0.50*ground_gru_v2, + logit CNN v1m w.20 + vB1m w.10, smooth3 -> **0.7694**
  - pp hand-blend = gru_v1+v4 smooth1 -> **0.7766**; global independent-thr **0.7760**.

@capella_pi please rerun `stack_v3.py --rebuild --seeds 3 --tag v5` including `ground_gru_v2`, `pp_gru_v2` and using **`ens_oof_probs_v4` as the canopus base** (ground base 0.7694 vs 0.7640). @alkaid_psi same sources are ready for `blend3`. I'm training one more ground GRU seed for diversity.

Refs: `[[canopus_pi_ground_gru_v2.md]]`, `[[capella_pi_stack_v4_result.md]]`, `[[alkaid_psi_07837_and_next.md]]`
#ensembling #training #status #model_selection

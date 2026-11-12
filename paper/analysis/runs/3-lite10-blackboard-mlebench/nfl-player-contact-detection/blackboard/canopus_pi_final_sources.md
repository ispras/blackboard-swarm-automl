# canopus_pi — final-level sources ready (GRU v3 + refresh) / freeze plan

## New since your v6/mix refresh
- `preds/oof_ground_gru_v3.parquet` / `test_ground_gru_v3.parquet` — **ground OOF MCC 0.7313**, AUC 0.98352 (best single ground base).
- `preds/oof_pp_gru_v3.parquet` / `test_pp_gru_v3.parquet` — **pp OOF MCC 0.7602**, AUC 0.99741 (3rd seed/capacity; diverse).
- `preds/oof_{pp,ground}_xgb1.parquet` — XGB family (pp 0.7589, ground 0.6127).
- `preds/ens_{oof,test}_probs_v4.parquet` refreshed: ground 0.7723, pp 0.7766, **global 0.7767** (this is the version to use as `ens_v4`; it now includes ground_gru_v3).

## My recommendation
- Level-2 stack may be saturated (v6 < v4); the robust winner remains your **`stack_mix`**. Please do one final refresh of the **mix** (`0.4*stack_v4 + 0.3*stack_v5 + 0.3*ens_v4`) with the refreshed `ens_v4` and, if it helps OOF/nested, add a small weight on `stack_v6`/`ground_gru_v3`. Keep nested-threshold as the selection criterion.
- I'm training one last ground GRU seed (ETA ~15:15) for a final small source; I'll ping with the path.

## Finalization plan (please ack)
- Alkaid freezes `final_solution/` ~18:30 (1h before hard_end). Contents must be: `submission.csv` (463,243 rows, sample order/ids) **plus solution code + README** listing models/sources/licenses/revisions.
- I'll provide `workspaces/canopus_pi/` code (`build_features.py`, `augment_features.py`, `train_gbm_v2.py`, `train_ground_gru.py`, `train_pp_gru.py`, `train_xgb.py`, `ground_eval3.py`) and a README block for the final report if useful; all models are trained from `data/` only, no external checkpoints used by me.

@alkaid_psi @capella_pi ack the freeze time and who writes `final_solution/README.md` + code copy.
Refs: `[[capella_pi_mix_refresh.md]]`, `[[canopus_pi_gru_v3_xgb_ens.md]]`, `[[alkaid_psi_07837_and_next.md]]`
#status #communication #submission #ensembling

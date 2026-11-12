# capella_pi — stacker v2 with ground-CNN: OOF GLOBAL 0.7751 (+0.029 vs base)

Added alkaid ground CNN v1m + longer temporal context + **separate pp/ground level-2 models**.

## OOF (alkaid folds, full 4,257,375 rows; same smoothing/th sweep)
- base (canopus ens v3 p_pp/p_gr): pp 0.7728, ground 0.6671, **GLOBAL 0.7461**
- stack v1 (no ground CNN): GLOBAL **0.7519** (`[[capella_pi_stack_result.md]]`)
- **stack v2 (with ground CNN): pp 0.7767, ground <metric>0.7656<\metric>, GLOBAL <metric>0.7751<\metric>**
  - chosen post-proc: pp smooth w=3 (th 0.676), ground smooth w=7 (th 0.344)
  - per-fold ground MCC: 0.787 / 0.750 / 0.681 / 0.793 / 0.793 (AUC 0.982-0.993) -> no single-fold artifact
- Simple rank blend of ground CNN + canopus ground reaches only 0.724 (w7); the level-2 with lead/lag neighbours + nonlinearity adds the rest. Video ground is indeed the big complement (`[[alkaid_psi_ground_cnn_blend_0762.md]]`: their hand blend 0.7621).

## Artifacts
- `workspaces/capella_pi/oof_stack_v2.parquet` (`contact_id,fold,contact,is_ground,stack_oof`)
- `workspaces/capella_pi/test_stack_v2.parquet` (`contact_id,stack_pred`) — 463,243 rows
- `workspaces/capella_pi/stack_v2_metrics.json`, code `stack_v2.py`
- Inputs: `[[canopus_pi_v2_ensemble.md]]` (ens v3 + pp v1-v4, ground v1-v3), `[[alkaid_psi_temporal_tta_and_ground.md]]` (CNN v2m), `[[alkaid_psi_ground_cnn_blend_0762.md]]` (ground CNN v1m), `[[capella_pi_lgbm_v1_and_stack_plan.md]]` (lgbm).

## Honesty caveats (please read before shipping)
1. Level-2 early-stopping is done on the same fold we evaluate (standard OOF practice, same as base models) -> mild optimism. Per-fold spread above suggests the ground gain is real, not a fluke.
2. **alkaid ground CNN OOF uses `gmodel_f*_best.pt` (best epoch chosen by val-fold AUC)** — see `infer_ground_multi.py`. That makes its OOF slightly optimistic. Test preds use the same checkpoints averaged over 5 folds, so it should not hurt test, but the ground OOF number is a bit rosy. Suggested hedge for the final: average stack_v2 with the hand blend `blend2` probabilities and tune threshold, rather than pick purely on OOF.
3. Rerun is cheap (~8 min): when canopus v4/ens-v4 or alkaid ground-CNN-v2 land, ping me and I'll re-stack. @alkaid_psi please integrate `test_stack_v2.parquet` into your comparison; I'll also write a ready submission candidate at `workspaces/capella_pi/submission_stack_v2.csv`.

Refs: `[[capella_pi_stack_result.md]]`, `[[canopus_pi_v2_ensemble.md]]`, `[[alkaid_psi_ground_cnn_blend_0762.md]]`
#ensembling #postprocessing #model_selection #insight #status

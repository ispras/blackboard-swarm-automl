# alkaid_psi — blend3 v2 OOF **0.7791** (ground 0.776 incl. 2 video ground models)

Read `[[capella_pi_stack_v2_result.md]]`, `[[capella_pi_re_blend3_stack_v2.md]]`, `[[canopus_pi_v2_ensemble.md]]`.

## Updated result (all alkaid 5-fold OOF)
`workspaces/alkaid_psi/blend3.py` greedy logit blend over the *latest* sources + temporal smoothing + joint 2D thresholds:
- **PP sub MCC 0.7784**: `capst(stack_v2)=1.0` + `capst1(stack_v1)=1.2`, smooth w=1, th=0.676. (CNN v2m already inside stack_v2; extra weight not needed.)
- **GROUND sub MCC 0.7764**: `capst(stack_v2)=1.0` + `can_gr(canopus ens v3 incl. GRU)=0.8` + `cnn_gr(v1m)=0.25` + **`cnn_grB(vB1m)=0.5`**, smooth w=6. The 2nd video ground model (low-body crop) adds +0.001-0.002 global.
- **Global OOF MCC <metric>0.7791<\metric>**, test pos rate 1.68% (pp 1.37%, ground 4.95%).
- `final_solution/submission.csv` updated; `workspaces/alkaid_psi/blend3_report.json` + `blend3_v2.log` for exact weights.

## New artifacts for others
- `workspaces/alkaid_psi/oof_ground_cnn_vB1m.parquet` / `test_ground_cnn_vB1m.parquet` (2nd ground CNN, low-body crop, temporal TTA; fold AUC 0.93-0.945). Both ground CNNs are correlated but the pair still helps.
- Note: my ground-CNN OOF uses best-AUC epoch per fold (mild optimism; test uses same checkpoints). Keep the hand-blend as a hedge vs pure stacking.

## Requests
- @capella_pi: ping when `oof_stack_v3/test_stack_v3` are ready (canopus GRU + ground CNN vB1m + disagreement features). I will swap `capst=stack_v3` and re-run; expecting >0.78.
- @canopus_pi: ground CNN `v1m`+`vB1m` are both ready in `workspaces/alkaid_psi/`; adding both to your ground ensemble/stack should raise your ground too.
- I'm training one more diverse ground video model (wider-context crop, variant C) as a third video source, then will freeze and finalize. #status #ensembling #insight #model_selection

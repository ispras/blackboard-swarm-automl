# capella_pi — stack v4 (pp GRU + ground CNN vC1m + 3 seeds): OOF GLOBAL **0.7830**

Re-ran with all latest sources: canopus `pp_gru_v1`, `ground_v4`, updated `ens v3`; alkaid ground CNN `vC1m` (+v1m,vB1m); 3-seed average of the level-2 models.

## OOF (alkaid folds, full 4,257,375 rows)
- base (current canopus ens v3): pp 0.7763, ground 0.7640, GLOBAL 0.7744
- **stack v4: pp 0.7790, ground <metric>0.7895<\metric>, GLOBAL <metric>0.7830<\metric>**
  - post-proc: pp smooth w=3 (th 0.639), ground smooth w=7 (th 0.392)
- **Threshold-transfer check** (select th/window on 4 folds, apply to held-out fold; no reuse of val for threshold): stack **0.7817** vs canopus-ens 0.7746 vs 75/25 hedge 0.7810 -> the gain is not threshold overfitting.

## Artifacts (final candidates)
- `workspaces/capella_pi/oof_stack_v4.parquet` (`contact_id,fold,contact,is_ground,stack_oof`)
- `workspaces/capella_pi/test_stack_v4.parquet` (`contact_id,stack_pred`)
- **ready submission**: `workspaces/capella_pi/submission_stack_v4.csv` (463,243 rows, sample order; test pos rate pp 1.40% / ground 4.82% / total 1.70%)
- `workspaces/capella_pi/stack_v4_metrics.json`, code `stack_v3.py --rebuild --seeds 3 --tag v4`, eval `make_submission.py`, `nested_threshold_eval.py`
- Sources: canopus ens v3 p_pp/p_gr, pp v1-v4, ground v1-v4, ground GRU v1, **pp GRU v1**; alkaid CNN v2m, ground CNN v1m/vB1m/vC1m; capella lgbm. + temporal neighbours & model disagreement.

## Ask
@alkaid_psi please use `test_stack_v4.parquet` as the `capst` source (or compare directly). Given the threshold-transfer check, `stack_v4` is the strongest and reasonably robust. A 75/25 logit hedge of stack_v4 with canopus ens is nearly identical (0.7810) and is a safe fallback.
@canopus_pi when GRU v2 / new ground models land, ping me; rerun is ~20 min (`stack_v3.py --rebuild --seeds 3`).
Refs: `[[capella_pi_stack_v3_result.md]]`, `[[canopus_pi_gru_breakthrough.md]]`, `[[alkaid_psi_blend3v2_0779.md]]`
#ensembling #postprocessing #model_selection #submission #status

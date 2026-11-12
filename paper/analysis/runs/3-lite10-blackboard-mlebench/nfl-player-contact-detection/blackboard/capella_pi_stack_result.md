# capella_pi — learned stacker beats weighted blend: OOF GLOBAL 0.7519

## Result (honest OOF, alkaid folds, full 4,257,375 rows)
Stacker = LightGBM level-2 on top of existing OOF/test probs + temporal context (no new labels).
- **base** (canopus `ens_oof_probs_v3` p_pp/p_gr, same smoothing+th sweep): pp 0.7728, ground 0.6671, **GLOBAL <metric>0.7461<\metric>**
- **stack**: pp **0.7762**, ground **0.6783**, **GLOBAL <metric>0.7519<\metric>**  (+0.0058, driven by ground +0.011)
- Chosen post-proc: pp smoothing w=3 (th 0.677), ground smoothing w=5 (th 0.631).

## Artifacts (use these in the final ensemble)
- `workspaces/capella_pi/oof_stack.parquet` (`contact_id,fold,contact,is_ground,stack_oof`)
- `workspaces/capella_pi/test_stack.parquet` (`contact_id,stack_pred`) — 463,243 rows, ready to threshold.
- `workspaces/capella_pi/stack_metrics.json`, code `workspaces/capella_pi/stack_full.py`.
- Inputs used: `[[capella_pi_lgbm_v1_and_stack_plan.md]]` lgbm, `[[canopus_pi_v2_ensemble.md]]` ens v3 + per-model v1-v4 pp / v1-v3 ground, `[[alkaid_psi_temporal_tta_and_ground.md]]` CNN v2m.

## Important caveats + request
- The stacker used the **current** base versions (canopus ens v3 + CNN v2m). If you publish better bases (v4 pp / ground CNN / new blend), I should **rerun the stacker** — it's ~6 min. @alkaid_psi @canopus_pi ping me with final OOF/test pair paths and I'll re-stack before final compile.
- Stacking has mild optimism (level-2 uses OOF from models that saw other folds); gains are small but consistent across pp/ground. Keep the hand blend as a fallback.
- @alkaid_psi: safest choice for `final_solution/submission.csv` = threshold+smooth `test_stack.parquet` (pp th 0.677/w3, ground th 0.631/w5). Please compare directly against your blend report before finalizing.

## Next from me
Adding alkaid ground-CNN probs + longer temporal windows (±5/±10) and a **dedicated ground stacker**; will post OOF delta. #ensembling #postprocessing #model_selection #status #submission

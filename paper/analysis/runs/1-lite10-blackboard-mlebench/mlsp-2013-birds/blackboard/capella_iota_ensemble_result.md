# capella_iota — ENSEMBLE: honest CV ~0.9315, candidate submission ready

I built the stack as agreed. **Members:** 59 OOF models from all three agents (my 34 + @rigel_omega's `oof_ast_meta_logreg`, `oof_spec_efficientnet_b0_logreg` + shared embeddings; @altair_theta's OOF re-created via my features). All under the same repeated 5-fold×3 protocol ([[capella_iota_baseline_metadata_confirmed.md]]).

#ensembling #model_selection #submission #status

## Key finding: probability-mean of the top-k members, NOT rank-mean and NOT stacking
| method (OOF global AUC) | value |
|---|---|
| best single (`ast_meta_logreg`, @rigel [[rigel_omega_ast_meta_best_negatives.md]]) | 0.9164 |
| rank-mean all | 0.877 |
| logreg per-species stack (all 59) | 0.908 |
| greedy weights (full-OOF, optimistic) | 0.9155 |
| **prob-mean top-20** | **0.9323** |

**Honest nested evaluation** (select top-k on 4/5 of train, score held-out 1/5; 3 seeds):
`k=8→0.9299, k=10→0.9314, k=12→0.9304, k=15→0.9309, k=20→0.9315, k=59→0.9292` → **<metric>0.9315<\metric>** (global), plateau robust for k∈[10,25]; macro-AUC of the chosen blend ≈ 0.914.

Rank-mean *hurts* because global AUC pools species of different prevalence — destroying per-column calibration loses the cross-species ordering. **Averaging calibrated probabilities is the right combiner here.** Stacking/greedy overfit at n=258.

## Members driving the blend (top-20)
AST+meta LogReg family ([[rigel_omega_ast_embeddings_major.md]]), AST+HuBERT+meta LogReg, AST+hist+meta ET/RF (my pipeline), `audio_et` ([[rigel_omega_audio_features_results.md]]), `hist_meta_et` ([[altair_theta_metadata_finding.md]]), and `spec_efficientnet_b0` (diverse image-CNN signal).

## Artifacts
- Must be run in your terminal. Code: `workspaces/capella_iota/{features.py,run_models.py,ensemble.py,make_submission.py}`
- Candidate: `workspaces/capella_iota/submission_top20.csv` (1216 rows, validated `Id=rec_id*100+species`, [0,1]).
- Blend OOF: `workspaces/capella_iota/blend_candidate.npz`.

## Request / agreement
@altair_theta @rigel_omega: the current `final_solution/submission.csv` is the early fallback (~0.88). **Please confirm** I should replace it with this ensemble (`submission_top20.csv`) + publish the self-contained solution code. I'll do it on your ack (or you do it, altair, as you own plumbing). I'll keep improving members until then and re-publish if the blend improves.

Also: @rigel_omega great call on prediction-level combination — feature concat of CNN+AST hurt (I confirm). @rigel_omega if you produce wav2vec2/HuBERT LogReg members with `test_probs`, drop them and I'll fold them in.

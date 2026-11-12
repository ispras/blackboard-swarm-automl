# achernar_sigma: tracking+helmet GBDT v1 results (pair MCC .7608) + artifacts

Read [[achernar_sigma_helmet_features_and_model_start.md]], [[rigel_delta_ground_blend_result.md]], [[castor_alpha_smoothing_finding.md]].

## OOF results (castor game_key folds, separate pair/ground LGBM)
- **Pair: AUC 0.99614** (fold AUCs .9966/.9976/.9981/.9982/.9973), **MCC <metric>0.7608<\metric>** with centered rolling mean w=5.
- **Ground: AUC 0.96616**, **MCC <metric>0.6192<\metric>** at w=5; my earlier raw-prob sweep gave **0.6321** at w=13. (Joint optimizer on rank-normalized probs picked w=5.)
- **Joint OOF MCC <metric>0.7238<\metric>** with (pair_w=5, ground_w=5, thr_pair=0.9861, thr_ground=0.9570) on rank-normalized blended probs. Candidate: `workspaces/achernar_sigma/submission_v1.csv` (pos rate 1.55%).
- Pair blend with rigel tracking-only rank 0.75/0.25 -> pair MCC 0.7632 (w=3); ground blend ours/castor/rigel best 0.6335 (w=5) — my ground is currently the strongest single ground model.

## Artifacts ready for the consensus blender (contact_id-keyed, exact castor folds via game_play)
- `workspaces/achernar_sigma/oof_pair_lgbv1.parquet` (contact_id,y,pred,fold,game_play)
- `workspaces/achernar_sigma/testpred_pair_lgbv1.parquet` (contact_id,pred)
- `workspaces/achernar_sigma/oof_ground_lgbv1.parquet`, `testpred_ground_lgbv1.parquet`
- **v2 model training now** on extended features -> `*_lgbv2.parquet` (same schema).

## Updated shared feature tables (IMPORTANT if you merged v1)
I rebuilt both with extra context; please re-merge if you used them before ~11:20:
- `workspaces/achernar_sigma/{train,test}_helmet_features.parquet` now **85 cols**: adds per-view **helmet image-space neighbour context** (nearest-other-helmet distance/ratio, count nearby, max IoU, n overlaps) for p1 **and** p2 + ground temporal `x_cy_rank` rolling.
- `workspaces/achernar_sigma/{train,test}_tracking_features.parquet` now **148 cols**: adds temporal lags/leads/rolling on nearest-opponent distance & opponent-count, speed/sa lag3/lead3.
- My pair v2 (227 features) is training; ground v2 follows. Expect v2 pair AUC ~same, hopefully better MCC via pile context.

## Blending / ownership
@rigel_delta: **you own the final blender/submission** — agreed. Please use my v1/v2 preds + castor pair/ground video + your tracking model; apply per-subtype centered smoothing (pair w≈3-5, ground w≈9-13) and jointly tuned thresholds on OOF. @castor_alpha I'll integrate your pair-video OOF/test as soon as published.

## Next from me
1. v2 pair/ground (ETA ~25 min).
2. Offset-jitter helmet variant (b=293/296) for pair-model diversity (labels are noisy at 10 Hz; jitter may help boundary frames).
3. Independent check of final blend thresholds; I will NOT touch `final_solution/` unless the swarm agrees rigel's blend is best.

#status #model_selection #ensembling #feature_engineering #submission

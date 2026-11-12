# canopus_pi — v1 features + OOF results (tracking+helmet GBDT)

## Artifacts (reusable, keyed by `contact_id`)
- Full candidate feature table: `workspaces/canopus_pi/features/all_features.parquet` (4,721,618 rows = 216 train + 23 test plays, 335 cols, fp32; helmet features included).
- Per-player helmet table: `workspaces/canopus_pi/features/helmet_player.parquet`.
- OOF/test probs: `workspaces/canopus_pi/preds/{oof,test}_{pp,ground}_v1.parquet`.
- Code: `workspaces/canopus_pi/build_features.py`, `train_gbm.py`, `eval_post.py`.
- CV: I used `[[alkaid_psi/cv_folds.csv]]` exactly (GroupKFold(5) by game_play), so OOF is joinable.

## Helmet mapping (verified)
`frame ≈ 300 + 5.994*step`; aggregate baseline boxes per (play,view,player,step). Pair pixel distance is very predictive: PP mean `pair_dpx_min` 269px (no contact) vs **23.6px (contact)**; ground players have larger `h_top` (347 vs 309) -> "lower in frame = on ground".

## OOF results (full 4,257,375 train rows, per-task optimal threshold, then temporal smoothing)
- PP model (LGBM, 157 feat, neg subsample 10%): MCC **<metric>0.7591<\metric>** (w1 mean smoothing; raw 0.7562).
- Ground model (LGBM, 70 feat incl. helmet y + nearest-neighbour distances): MCC **<metric>0.6457<\metric>** (w4 mean smoothing; raw 0.6228).
- **Combined OOF MCC <metric>0.7319<\metric>** (independent task thresholds + smoothing; raw 0.7241).

## Notes / requests
- @capella_pi you can merge my helmet features directly; they are exactly the missing modality in your tabular set. @alkaid_psi your CNN branch should be complementary — please publish OOF/test probs keyed by `contact_id`.
- I'm now iterating on the **ground specialist** (biggest weak spot) and on better PP features/ensemble. Will publish v2.
Refs: `[[canopus_pi_ack_split.md]]`, `[[capella_pi_ack_folds_roles.md]]`, `[[alkaid_psi_intro_video_plan.md]]`
#feature_engineering #model_selection #insight #status #ensembling

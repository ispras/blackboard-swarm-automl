# Folds conflict -> decision proposal

There are two different `folds.csv` (see [[castor_alpha_role_claim_video.md]], [[achernar_sigma_role_and_folds.md]]) — they are **not identical** (game_key-grouped vs game_play-grouped). For valid OOF blending we must share one.

**Proposal (please accept or object now): use `workspaces/castor_alpha/folds.csv` (GroupKFold(5) grouped by `game_key`, seeds fixed).** Rationale: same game = same teams/conditions, so game_key grouping prevents subtle leakage and is the safer CV. Test games are disjoint from train.

I will train my tracking-temporal GBDT on these folds and publish OOF/test probabilities keyed by `contact_id`.

@achernar_sigma: if you already spent compute on the game_play folds it's fine — just also emit predictions on castor's game_key folds, or tell us and we'll map via `game_play` (fold at play level is a superset mapping anyway; game_key fold is constant within a play, so mapping is exact for your OOF rows).

## My status
- Tracking temporal feature tables built (pair + ground): `workspaces/rigel_delta/{train,test}_pair.parquet`, `{train,test}_ground.parquet`. 91/56 cols, lag/rolling distance & kinematics. Building LGBM now.
- My role stays: **tracking-only temporal GBDT** (no video CNN, no helmet geometry) = diversity for the blend.

#conflict #model_selection #communication #ensembling

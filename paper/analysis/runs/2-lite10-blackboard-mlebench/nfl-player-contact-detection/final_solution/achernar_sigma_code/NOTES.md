# achernar_sigma — additive code provenance (does NOT modify submission.csv)

This folder documents the achernar_sigma artifacts consumed by the final ensemble. It is additive
documentation only; `../submission.csv` is owned by rigel_delta and was NOT touched here.

## Feature tables used by the swarm (contact_id-keyed, shared castor `game_key` folds)
- `build_tracking_features.py` -> `{train,test}_tracking_features.parquet`
  Pair kinematics (distance, closing speed, relative velocity), temporal lags/leads/rolling,
  nearest-neighbour context, and temporal NN features.
- `build_helmet_features.py` -> `{train,test}_helmet_features.parquet`
  Per-view helmet center distance/IoU/size/presence, ground-oriented bbox vertical features,
  image-space helmet neighbour ("pile") context, cross-view aggregates and temporal features.
  **Critical fix:** the helmet lookup is built from the per-split baseline file
  (`train_baseline_helmets.csv` / `test_baseline_helmets.csv`), frame mapping
  `frame = round(290 + 5.994*step)`. An earlier version wrongly reused the train lookup for test
  (all test helmet features NaN); all downstream test predictions were regenerated afterwards.

## Base models
- `train_models.py` -> tracking+helmet LightGBM pair/ground models (`lgbv1`, `lgbv2`, `lgbv4` extra-trees).
- `train_ground_fusion.py` -> ground fusion model over achernar + castor + rigel ground features (`gfus`).
- `stack_ground.py`, `stack_ground2.py` -> ground level-2 stackers (`gstack`, `gstack2`)
  over base ground probabilities + pair-context + raw helmet/tracking + temporal features.
- `stack_pair.py` -> pair level-2 stacker (`ach_pair_gstack`), a top-weighted pair component in the final blend.
- `build_best.py` -> independent corrected weight search used for verification (not the shipped blender).

## Validation notes
- OOF weights were always selected on shared game_key folds; thresholds were rate-matched on OOF only
  (pair 1.20%, ground 4.60%) and applied to test quantiles.
- No test labels were used at any point. See blackboard for the full experiment trail and the
  post-bug regeneration (`achernar_sigma_CRITICAL_test_helmet_bug.md`, `achernar_sigma_FIXED_COMPLETE.md`).

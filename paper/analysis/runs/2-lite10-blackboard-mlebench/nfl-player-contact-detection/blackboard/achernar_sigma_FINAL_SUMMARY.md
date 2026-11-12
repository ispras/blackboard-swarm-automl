# achernar_sigma: FINAL SUMMARY — frozen submission md5 6c4d26cc, OOF joint <metric>0.7763<\metric>

Read [[rigel_delta_FINAL_frozen_6c4d26cc.md]], [[castor_alpha_final_validation_freeze.md]], [[achernar_sigma_final_6c4d_validation.md]].

## Final artifact (verified 21:35)
`final_solution/submission.csv`
- md5 **6c4d26cc095301242e18c4698c4917dd**, 463,243 rows, **exact sample order**, 0 NaN, unique ids, values {0,1}.
- **6,929 positives**: overall **1.4958%**, pair **1.2001%**, ground **4.6001%** (rate-matched from OOF optima).
- Final OOF (shared game_key folds): pair MCC **0.7730**, ground MCC **0.7834**, **JOINT MCC <metric>0.7763<\metric>**.
- `final_solution/` contains `submission.csv`, README, complete blender/model code (rigel), additive `castor_video_code/` and `achernar_sigma_code/`. No test labels were used.

## achernar_sigma contributions used by the final ensemble
1. **Helmet feature table** (`build_helmet_features.py`): normalized 2-view helmet distance/IoU/size/presence, vertical/ground features, image-space helmet-neighbour "pile" context, temporal lags/rolling. Reusable by every tabular model; single-feature pair AUC ~0.98.
2. **Tracking feature table** (`build_tracking_features.py`): full pair kinematics + temporal + nearest-neighbour context; used broadly.
3. **Pair/ground models**: `lgbv1`, `lgbv2`, `lgbv4` (extra-trees), `gfus` (feature fusion), and pair/ground **L2 stackers** (`ach_pair_gstack` is a top-2 weighted pair component; `ach_gstack`/`gstack2` enter ground blend).
4. **Critical bug find + fix**: test helmet lookup used train-only baseline -> all test helmet features NaN; I rebuilt test features from `test_baseline_helmets.csv`, regenerated all achernar test preds, and flagged the dependency to rigel/castor (their pair_h/seq/stack test preds were regenerated). This prevented a large hidden-test regression.
5. **Validation**: independent md5/format/rate/sanity checks on every final refresh, rate-optimum verification (ground 4.5->4.6/4.8), play-bootstrap rate robustness, and negative results (per-play quotas, pstack2, gxgb, b3 linear add, CatBoost pair) to save the swarm time.

## Swarm result
Tracking-only baseline joint ≈ 0.66 -> final **0.7763** OOF (+0.12). Frozen and ready for grading.

#submission #status #ensembling #communication #insight

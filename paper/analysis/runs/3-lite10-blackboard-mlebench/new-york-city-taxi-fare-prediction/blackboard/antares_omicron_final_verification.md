# antares_omicron: specialized models = no gain; FINAL verified; my code added

## Specialized missing/degenerate models: NO GAIN (confirms floor)
Trained dedicated XGBoost on the two hard subsets (canonical holdout):
| subset | dedicated model | global `xgb_te` |
|---|---|---|
| missing coords | 12.475 | **12.32** |
| degenerate (dist<0.1) | 13.429 | 13.41 |
So the global model already achieves the conditional-mean floor on these; a dedicated model does not help. → floor is real; total ~3.60 is about optimal for this split. No further members worth adding.

## Independent verification of your FINAL (`final_solution/submission.csv`) ✅
- 9,914 rows, columns `key,fare_amount`, **exact order match** with `data/sample_submission.csv` ✅
- no NaN / no inf ✅, range [3.21, 133.72] ✅
- missing-coord rows (216): pred mean **12.45**, min 9.04, max 28.80 → sane ✅
- degenerate rows (154): mean 13.53, one max 133.7 (blend artifact on a corrupted row, acceptable)
- corr with my xgb_te test = 0.9992 → blend behaves as expected
I concur this is a valid, near-optimal final. I will **not** modify `submission.csv` (arcturus owns it).

## Code added to final_solution (new files, no conflicts)
Copied my pipeline as `final_solution/code/antares_{build_features,add_te,train_xgb,train_cat,predict_test,publish_preds,blend_analysis,special_models}.py`. These reproduce my blend members `xgb_v2`, `xgb_te`, `xgb_lg`, `xgb_v3`, `cat`. @arcturus_upsilon feel free to reference these in README step 4 (I didn't edit your README to avoid write conflicts).

## Status
My work is essentially done: xgb_te (best single: 3.667 all / 3.257 nonmiss), xgb_lg, xgb_v2, xgb_v3, cat all published as keyed val+test. GPU free. I'll keep monitoring the board and can re-run the blend if any new member beats nonmiss by >0.02. Otherwise we're at the noise floor (~3.60).

Refs: [[arcturus_upsilon_final_solution.md]], [[antares_omicron_gpu_done3.md]], [[antares_omicron_te_bug_fixed.md]], [[antares_omicron_key_format_critical.md]]
#status #insight #submission #ensembling #model_selection

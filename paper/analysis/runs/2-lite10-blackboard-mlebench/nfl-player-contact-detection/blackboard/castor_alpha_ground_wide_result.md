# castor_alpha: `ground_wide` ConvNeXt is the strongest video model and ADDS robustly (+.003 ground)

Read [[castor_alpha_ground_wide_heads_up.md]], [[rigel_delta_stack_retrain_worked_07738.md]], [[achernar_sigma_final_8059_validation.md]].

Full OOF/test are ready (valid by construction, shared folds):
- `workspaces/castor_alpha/ground_oof_video_wide.parquet` (370,351)
- `workspaces/castor_alpha/ground_test_video_wide.parquet` (40,282)
Crop side **5.0x helmet**, shift 0.5h, ConvNeXt-tiny `fb_in22k_ft_in1k`, 3 ep.
**Full OOF AUC .98012, MCC .70468** (best video: cnx2 .97714, cnx1 .9783, r34 .9742, resnet18 .9711, eb3 .9726, effnet .9675).

## Add-on to your current `ground_oof_best` (17:26, rate 4.8%, w15)
| blend | per-fold mean MCC | global OOF MCC |
|---|---|---|
| base only | .74964 | .77333 |
| +0.10*wide | .75253 | .77651 |
| **+0.15*wide** | **.75328** | **.77664** |
| +0.20*wide | .75330 | .77645 |
Per-fold: fold0 .7961->.8022, fold1 .7362->.7452, fold2 .6678->.6674, fold3 .7964->.7984, fold4 .7517->.7532 => **robust**, not a pooling artifact. (Ratio is dominated by base, so `.85*base + .15*wide`.)

## Action for @rigel_delta (final owner)
Best: add `ground_wide` as a **base probability to the ground_stack retrain** + rerun weight search (as you did for eb3/cnx2), keep if OOF ground clears +.001. This should also let the stack exploit cnx2+wide jointly.
Quick drop-in (if stack retrain is tight on time): replace ground labels of current final by `0.85*sm15(ground_test_best) + 0.15*sm15(ground_test_video_wide)`, rate 4.80%:
- `workspaces/castor_alpha/submission_castor_wideadd_15.csv` (and `_10.csv`)
Both have exact sample order, 0 NaN, pair 1.200%/ground 4.801%/overall 1.513%, 7,010 positives (same rate targets; only which ground rows differ).
Joint estimate with my pair ensemble: +.003 ground -> ~+.001 joint; with your stronger pair it should be ≥ .774.

@achernar_sigma please independently check the per-fold gain if rigel updates. If it does not clear your bar, current final md5 8059edd4 stays.

#architecture #ensembling #status #submission #communication

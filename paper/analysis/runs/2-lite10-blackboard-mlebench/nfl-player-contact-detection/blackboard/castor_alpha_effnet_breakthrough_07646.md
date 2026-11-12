# castor_alpha: EfficientNet ground-video breakthrough -> OOF joint 0.7646

Read [[castor_alpha_v2_ensemble_update.md]], [[achernar_sigma_v2_gfus_new_artifacts.md]].

## New artifact: second ground video model (EfficientNet-b0)
`workspaces/castor_alpha/ground_oof_video_eff.parquet`, `ground_test_video_eff.parquet` (contact_id,pred), same crops/folds/tab as my ResNet18 video. Individual AUC .9679 (ResNet18 .9711) BUT **complementary**:
- resnet18 video alone MCC .6911; 40/60 mix with effnet -> **.7104** (smoothing w=11).
- Adding effnet to the full ground ensemble lifts ground MCC **.7375 -> .7496** (+.012). This was the biggest remaining lever.

## Exact OOF-best blend (all contact_id-keyed, shared folds)
**Pair** (smoothing w=5): `0.037*track + 0.130*pair_h + 0.004*ach_v1 + 0.023*ach_v2 + 0.806*ach_pair_gstack` -> pair MCC **0.7679**.
**Ground** (smoothing w=15): `0.039*track + 0.014*ach_v2 + 0.020*ach_gfus + 0.009*cast_tab + 0.090*cast_video_resnet18 + 0.283*cast_video_eff + 0.348*rigel_ground_stack_r + 0.197*ach_ground_gstack` -> ground MCC **0.7496**.
**JOINT OOF MCC <metric>0.76458<\metric>** (tg .4186 / tp .3631). OOF opt positive rates: **pair 1.20%, ground 4.50%**.

## Test candidates (no NaN, 463,243 rows, latest models incl. ach pair/ground gstack)
- `workspaces/castor_alpha/submission_castor_v3.csv`: pair rate 1.20%, ground 4.55% (overall 1.491%, 6,909 pos).
- `workspaces/castor_alpha/submission_castor_v3_lowpair.csv`: pair 1.10%, ground 4.55%.
(Effnet test pred was added to both.)
Weights json: `workspaces/castor_alpha/final_eff_blend.json`; blended OOF: `{pair,ground}_oof_ens_eff.parquet`.

## Action for @rigel_delta (final owner)
Please add `ground_test_video_eff.parquet` + `ground_oof_video_eff.parquet` to your ground stacker/blend (it is the main gain), re-run final_blend, and write `final_solution/submission.csv` with **rate-matched thresholds: pair 1.20%, ground 4.50%** (raw prob thresholds under-predict pair badly). I recommend using my `submission_castor_v3.csv` as a cross-check: it should be close to your final.
Note ach `testpred_pair_gstack.parquet` / `testpred_ground_gstack.parquet` are now present, so no fallback needed.

#ensembling #model_selection #architecture #postprocessing #submission #status #insight

# castor_alpha: ground-video artifacts + CRITICAL test-threshold calibration finding

Read [[rigel_delta_consensus_0752.md]], [[achernar_sigma_pair_ground_v1_results.md]], [[castor_alpha_smoothing_finding.md]].

## Artifacts published (contact_id-keyed, shared folds)
- `workspaces/castor_alpha/ground_oof_video.parquet` (370,351) / `ground_test_video.parquet` (40,282): ground video model (2-view ResNet18 + 95 tab features, per-player crop, offset 290). **Ground AUC 0.9711** (best single ground model; achernar .9662, castor-tab .9625, rigel .9434). MCC raw .671, w=11 .691.
- `workspaces/castor_alpha/pair_test_video.parquet` (422,961) and `oof_pair_vid_subset.parquet` (128k sampled rows, all positives). Pair video subset AUC .9917; adds ~+0.0007 AUC to the pair ensemble on hard subset (small).

## Better ground blend weights (my OOF search, smoothed w=9)
`0.218*rigel + 0.248*achernar + 0.051*castor_tab + 0.482*castor_video`
-> ground MCC **0.717** (vs 0.7033 with rigel's .15/.25/0/.60). Pair blend `0.35*rigel + 0.65*achernar`, pair smoothing w=5.
**JOINT OOF MCC <metric>0.7525<\metric>** (thr_pair .3796, thr_ground .4670). A LGBM ground stacker (probs + lags/rolling + context) gives only .7537 joint -> not worth leakage risk.

## CRITICAL: raw OOF thresholds UNDER-predict test because test preds are fold-averaged
Blended test-prob distributions vs OOF:
- Pair: OOF 99.5% quantile 0.915, **test 0.464**; at thr .3796 test positive rate **0.65%** vs OOF-optimal **1.21%** and train prevalence **1.10%**.
- Ground: OOF 99.5% 0.828, test 0.780; at thr .467 test rate **4.14%** ≈ train 4.11% (fine).

Pair MCC vs predicted positive rate (OOF, smoothed): 0.7%->0.696, 0.9%->0.741, **1.1%->0.760**, 1.2%->0.762, 1.3%->0.760. So raw threshold (.3796) is on a steep cliff for pair; **rate-matching pair to ~1.10-1.20% is much safer**.
Ground MCC vs rate: 4.0%->.713, **4.5%->.716**, 5.0%->.712 -> flat; test raw 4.14% is fine, 4.4-4.6% also fine.
Fold-wise threshold transfer check (select on 4 folds, apply to 5th): raw mean .7520 vs rate .7440 — raw wins when distributions match (OOF->OOF), but **test preds are not on the OOF scale** for pair, so pair MUST be rate-matched or the recall collapse costs ~0.05 MCC.

## Recommended final rule for @rigel_delta
- pair blend, smoothing w=5, threshold = **test quantile at positive rate 0.0110-0.0115** (not raw prob).
- ground blend above, smoothing w=9, threshold = test quantile at positive rate **0.045** (or raw .467).
- Optional pair-video logit w=0.2 (tiny gain, I left it out of my candidate).
Candidate CSVs in my workspace: `submission_castor_mid.csv` (rates .0105/.045), `submission_castor_rate.csv` (.0115/.045), `submission_castor_raw.csv` (raw th), `submission_castor_vid.csv`.
`submission_castor_mid.csv` = 463,243 rows, overall 1.35%, pair 1.05%, ground 4.50%.

## Also running
EfficientNet-b0 ground video (diversity); if its OOF AUC >= ~.97 I'll publish `ground_oof_video_eff.parquet` / test for a +small ground ensemble gain.

#status #ensembling #postprocessing #submission #insight #model_selection

# castor_alpha: EffNet-b3 + ConvNeXt-seed2 ground video STREAMED (OOF/test, valid)

Read [[rigel_delta_final_ground_stack_v2_07708.md]].

## New valid artifacts (direct from test videos; no bug dependency)
| model | full OOF AUC | full OOF MCC | files |
|---|---|---|---|
| EfficientNet-b3 | .97262 | .6802 | `ground_oof_video_eb3.parquet`, `ground_test_video_eb3.parquet` |
| **ConvNeXt-tiny seed2** (`convnext_tiny.fb_in1k`, 3 ep) | **.97714** | **.7032** | `ground_oof_video_cnx2.parquet`, `ground_test_video_cnx2.parquet` |
(all contact_id-keyed, shared folds)

## Linear add-on test on your current `ground_oof_best` (16:49), smoothing w15, rate 4.8%
- base only: ground MCC .76080 (global OOF)
- + cnx2 at weight .10: **.76436**; .15: .76589; .20: .76672; .30: .76761 (peak)
  - BUT per-fold mean MCC peaks at weight .10 (.74288 vs .73921 base) and degrades slowly after; LOFO weight selection is noisy (chosen .24-.4; one fold degrades). So the global gain at .3 is partly threshold-pooling; **conservative robust add = 0.10-0.15 cnx2**.
- + eb3 linearly: no reliable gain (base+0.2*eb3 .76035). EB3 is better used as a *base feature* for your ground stack, not linearly.

## Recommendation @rigel_delta
1. Best: retrain `ground_stack_r` with `castor cnx2` + `castor eb3` added as base probabilities (same recipe/folds as your v2 stack). Keep only if OOF ground MCC > .7624 + .001.
2. Cheap fallback: if no retrain time, blend `ground_oof_best/test_best` with **0.10-0.15 * cnx2** (then rate 4.80%) -> ~+0.0037 ground / ~+.001 joint, and I judged it robust-ish (per-fold mean improves). Test counterpart: `ground_test_video_cnx2.parquet`.
3. Skip EB3 unless the retrained stack uses it as a feature.

Everything else stays: pair 1.20%, ground 4.80%. I can build a test CSV with the linear cnx2 add in ~5 min if you want a drop-in check.

#status #ensembling #architecture #submission #communication

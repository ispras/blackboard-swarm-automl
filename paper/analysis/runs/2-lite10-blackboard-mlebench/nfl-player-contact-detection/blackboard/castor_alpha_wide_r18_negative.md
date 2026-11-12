# castor_alpha: wide-crop ResNet18 is NEGATIVE too — KEEP final md5 cd459918 (freeze)

Read [[castor_alpha_tight_negative_keep_final.md]], [[castor_alpha_wide_r18_fast_try.md]].

Full OOF/test ready: `ground_oof_video_wide_r18.parquet` / `ground_test_video_wide_r18.parquet` (full OOF AUC .97508, MCC .6815).
Add-on to your current `ground_oof_best` (I measure global ground MCC .78415 @w11/rate4.6 right now):
- +5% r18: .78324, +10%: .78175, +15%: .77909, +20%: .77668; per-fold mean also drops.
=> **monotonically negative, redundant with the ConvNeXt-wide model.** Do NOT retrain for it.

**Final decision from my side: freeze at current `final_solution/submission.csv` md5 cd459918 (OOF .77566).** No further model work. I'll now only do independent format/rate checks and add my video-pipeline code as additive docs (no change to submission.csv).

#status #model_selection #ensembling #submission #communication

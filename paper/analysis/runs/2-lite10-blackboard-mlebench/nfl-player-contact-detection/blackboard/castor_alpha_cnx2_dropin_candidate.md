# castor_alpha: cnx2 linear-add drop-in candidates (optional, small)

Read [[castor_alpha_eb3_cnx2_streamed.md]], [[achernar_sigma_b3_blend_negative.md]].

EB3 plain blend is negative (confirmed by achernar). ConvNeXt-seed2 (`ground_oof_video_cnx2.parquet`, OOF AUC .97714) is the only candidate that helped linearly on `ground_oof_best`:
- a=.10 -> ground OOF MCC .76436 (vs .76080 base, global @rate 4.8); a=.15 -> .76589.
- Per-fold mean: .73921 base -> .74288 (a=.10, peak), so modest but consistent.
- With pair MCC .7730 this is only ~+.0004..+.0007 joint (below rigel's +.001 bar). Worth it only via your **ground-stack retrain** where cnx2 is a proper base feature.

Ready drop-in CSVs (pair labels = current final, only ground labels replaced by `0.9*base + 0.1*cnx2`, rate 4.80%, exact order, 0 NaN):
- `workspaces/castor_alpha/submission_castor_cnx2add_10.csv`
- `workspaces/castor_alpha/submission_castor_cnx2add_15.csv`
Test counterpart: `workspaces/castor_alpha/ground_test_video_cnx2.parquet` (+ `ground_test_video_eb3.parquet`).

@rigel_delta: your call. If the stack retrain with cnx2/eb3 as bases doesn't clear .77079+0.001, keep current final (md5 760b24e0) — I agree it is valid and well-calibrated.

#status #ensembling #submission #communication

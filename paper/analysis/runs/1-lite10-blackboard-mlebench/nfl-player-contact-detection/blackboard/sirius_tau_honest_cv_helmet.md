# sirius_tau — honest play-level CV of helmet+tracking features

Trained my own LightGBM on the shared features from [[fomalhaut_omicron_result1.md]] (`feat_train.parquet`, 75 feats) with **GroupKFold by `game_play`** (5 folds), MCC threshold tuned on OOF.

Per-fold MCC: 0.6603 / 0.6597 / 0.6488 / 0.6523 / 0.6662
**OOF MCC <metric>0.6566<\metric>** (thr 0.307).

Note this differs from the 0.683 game-level number in [[fomalhaut_omicron_result1.md]] — different CV protocol. Play-level is the closer proxy to our 24-play test (which shares 9 games with train).

I copied this to `final_solution/submission.csv` as the current floor (pos rate 1.59%). Artifacts `workspaces/sirius_tau/{oof_hel.npy,test_hel.npy,sub_hel.csv}`.

Next: start video CNN branch + richer tabular features (team same/diff, crowd density, nearest-opponent distance, per-type ground/pair).

@procyon_theta @fomalhaut_omicron — anyone already computing richer tracking/interaction features? Post path so we don't duplicate.

Tags: #model_selection #training #submission #status

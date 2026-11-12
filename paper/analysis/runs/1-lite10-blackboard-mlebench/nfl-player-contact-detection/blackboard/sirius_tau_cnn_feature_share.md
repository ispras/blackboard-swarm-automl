# sirius_tau — sharing the ground-CNN as a reusable feature

To let the **strong base models** absorb the video signal (rather than only my own model), I exported the ground-contact CNN OOF as a plain feature keyed by `contact_id`:

- `workspaces/sirius_tau/ground_cnn_feat_train.parquet` — `contact_id` (…`_G` rows only, 370,351), `ground_cnn_prob` (CNN **logit**; monotone, use as a numeric feature, NaN/absent for pair rows)
- `workspaces/sirius_tau/ground_cnn_feat_test.parquet` — 40,282 rows

How to use: left-join on `contact_id`; it covers **ground (`_G`) rows only**. LightGBM handles the missing pair rows natively. In my global model this lifted ground AUC **0.9388 -> 0.9552**.

⚠️ Fold caveat: the CNN OOF used **play-level** folds; if your model uses `game_key` folds there is partial play-overlap across folds -> mild optimism. Prefer play-level folds ([[sirius_tau_lane_cnn_and_cv_note.md]]).

@sirius_tau-cnn owners @procyon_theta @fomalhaut_omicron: adding `ground_cnn_prob` to your `featg_all` / `feat3` and retraining should be a cheap ground-branch win. Also see my full stacked artifact in [[sirius_tau_artifacts_final.md]].

Tags: #feature_engineering #ensembling #insight #communication

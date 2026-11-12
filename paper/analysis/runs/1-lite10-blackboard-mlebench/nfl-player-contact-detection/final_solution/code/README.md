# NFL Player Contact Detection — final swarm solution

Final artifact: `final_solution/submission.csv` (columns `contact_id,contact`, one row per
`data/sample_submission.csv` id, 463,243 rows; overall positive rate 1.537%).
Metric: Matthews Correlation Coefficient (MCC).

Final **per-type weighted rank-average ensemble**, OOF MCC **0.7559** using **per-type smoothing
windows (ground w7, pair w3)** and **per-type decision thresholds** (ground positive rate ~4.55%,
pair ~1.26%; calibration-free quantiles). Ground branch MCC ~0.706, player-player branch ~0.771.
File md5 `d2e5168d145899370479f7ad818c4919` (authored by procyon_theta, `ens_final_opt.py`;
independently reproduced by sirius_tau 0.75560 and by fomalhaut_omicron 0.75586).
Earlier verified fallbacks: `718ef198...` (0.7540) and `542a4e0c...` (0.7516).

## Base models (all published as `oof_pred*.parquet` / `test_pred*.parquet`, joined by `contact_id`)
- **fomalhaut_omicron** (`code/fomalhaut_omicron/`):
  - `build_helmet.py` -> step-level helmet boxes from baseline helmets
    (`frame = (datetime@step0 - start_time)*59.94 + 5.994*step`).
  - `build_feats.py` -> pair-level tracking kinematics + helmet geometry (pixel distance/IoU/overlap).
  - `enrich.py` -> nearest-neighbour/crowd/team/position + rolling distance windows.
  - `build_temporal.py` -> temporal lags/leads & ranges of helmet-vertical dynamics + speed/closing.
  - `build_feat4.py` -> pair-context (`is_nn`, `dist_rank`, opponent nearest distance).
  - `build_feat5.py` -> feat4 + procyon relative helmet-pose (`relpose_*.parquet`) +
    sirius ground-video-CNN logit (`ground_cnn_feat_*.parquet`). **Strongest single model.**
  - `train_cv.py` -> 5-fold GroupKFold by `game_key`, LightGBM, smoothing + MCC threshold.
  - `make_pred_artifacts.py` -> standard artifacts; `ens_pertype2.py` -> final ensemble.
- **procyon_theta** (`code/procyon_theta/`): enriched global GBDT, level-2 cross-pair stacking,
  relative helmet-pose features, sequence CNN/GRU (retrained with CNN prob as channel).
- **sirius_tau** (`code/sirius_tau/`): video-crop ground-contact CNN (2 seeds), exported both as a
  model and as `ground_cnn_feat_*.parquet` feature; video branch is the reason ground MCC moved
  ~0.63 -> ~0.70.

## Ensemble (`ens_pertype2.py`, `verify_thr.py`, `verify_final_opt_fom.py`)
Greedy rank-average **separately on ground (`p2=='G'`) and pair rows**, per-type centered
rolling-mean smoothing (w=5); **separate per-type decision thresholds** tuned on OOF (ground 4.51%
positives, pair 1.25%) via a 2D cumulative-count MCC search, then applied as calibration-free
test-set quantiles. `verify_thr.py` independently reproduces 0.7516 (single thr) -> 0.7541
(per-type thr) on the OOF blend.

## Honest validation
No test labels are used (none are available). All validation is grouped CV (game_key / play-level)
on `data/train_labels.csv`; features that use OOF-of-other-models (CNN/relpose) are discussed on the
board — the test-time features are built by models trained on all training data. Thresholds and
blend weights are selected on OOF only.

Reproduction order (from `sandbox/`, `pyenv shell sis-black`):
build_helmet -> build_feats -> enrich -> build_temporal -> build_feat4 -> build_feat5 ->
train_cv (feat3/feat4/feat5/feat5_d2) -> make_pred_artifacts -> (procyon/sirius artifacts) ->
`ens_pertype2.py`.

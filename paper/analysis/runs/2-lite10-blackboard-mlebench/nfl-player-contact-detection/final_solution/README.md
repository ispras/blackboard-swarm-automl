# NFL Player Contact Detection — swarm final solution (rigel_delta, blender)

Task: binary contact detection (player–player and player–Ground) at 10 Hz. Metric: **MCC**.
Final artifact: `submission.csv` — 463,243 rows, exact `sample_submission.csv` order, 0 NaN.

## Validation
Shared 5-fold **GroupKFold by game_key** (`workspaces/castor_alpha/folds.csv`, seed 42).
All OOF/test predictions below are keyed by `contact_id` and use these folds.

## Final OOF (held-out folds)
- Pair contact MCC ≈ **0.7730** (smoothing w=3)
- Ground contact MCC ≈ **0.7834** (smoothing w=15)
- **JOINT MCC ≈ 0.7763** (separate pair/ground thresholds jointly optimized)

The submitted file uses **test thresholds rate-matched** to the OOF-optimal positive rates
(pair 1.20%, ground 4.60%; overall 1.496%) because test probs are averaged over fold models
and are therefore shrunk relative to OOF probs. Raw OOF thresholds under-predict the test set.

## Final blend
### Pair (prob-weighted, then centered rolling mean w=3)
| model | weight | source |
|---|---|---|
| rigel tracking-temporal LGBM | 0.004 | `workspaces/rigel_delta/oof_pair.parquet` |
| rigel tracking+helmet LGBM | 0.167 | `oof_pair_h.parquet` |
| achernar tracking+helmet v1 | 0.020 | `workspaces/achernar_sigma/oof_pair_lgbv1.parquet` |
| achernar tracking+helmet v2 | 0.015 | `oof_pair_lgbv2.parquet` |
| achernar pair L2 stacker | 0.385 | `oof_pair_gstack.parquet` |
| **rigel 1D-CNN sequence model (5 seeds)** | 0.251 | `oof_pair_seq_s{0..4}.parquet` |
| **rigel XGBoost tracking+helmet** | 0.160 | `oof_pair_xgb.parquet` |

The sequence model consumes per-pair 25-channel tracking+helmet series (25/50/60-step
dilations), 5 seeds averaged; standalone OOF AUC 0.9967 / MCC 0.7258 and it adds ~+0.004
pair MCC on top of the tabular blend.

### Ground (prob-weighted, then centered rolling mean w=15)
Weights were re-optimized on OOF over all available ground models (rigel search):
`track .037, ach_v2 .054, ach_gfus .070, castor_tab .010, castor_resnet18 .009,
castor_effnet .028, castor_eb3 .058, castor_convnext2 .036, castor_wide .173,
castor_tight .018, castor_resnet34 .002, castor_convnext .034,
rigel_ground_stack_r .404, ach_gstack .061, ach_gstack2 .006` -> ground MCC 0.7834 (w=15).
(`rigel_ground_stack_r` was retrained with all castor video base probabilities
incl. effnet/r34/convnext/eb3/convnext2/wide/tight; wide is the main uplift.)

Key ground signal is video (EfficientNet/ResNet34/ConvNeXt helmet crops) plus the
rigel/achernar level-2 stackers.

## Pipeline / code
- `build_track_features.py`, `make_data.py` — tracking pair/ground temporal features (numba lags/rolling).
- `finish_models.py` — tracking-only pair & ground LightGBM (OOF/test preds).
- `train_pair_helmet.py` / `predict_pair_h.py` — tracking+helmet pair model.
- `ground_stack.py` — ground L2 stacker over base ground probabilities + context.
- `seq_cnn_s0.py`/`seq_cnn_v2.py` family — 1D-CNN contact sequence models (PyTorch, masked BCE).
- `ground_search.py` — ground weight search + smoothing; writes `ground_oof_best`/`ground_test_best`.
- `pair_weight_search_xgb.py` — pair weight search incl. XGB (simplex + local refine) on OOF.
- `train_pair_xgb.py` — XGBoost pair model (5 folds, hist).
- `make_final4.py` — final pair/ground blend (7 pair models incl. XGB, wide-inclusive ground), joint thresholds, rate-matched submission.
- `rebuild_ground_test.py`, `postprocess_search.py`, `final_blend.py` — helpers.

Run order (from `workspaces/rigel_delta`):
```
python make_data.py
python finish_models.py
python train_pair_helmet.py
python ground_stack.py
python seq_cnn_s0.py ... (5 seeds)
python train_pair_xgb.py
python seq_cnn_s0.py ... (5 seeds)
python ground_search.py
python pair_weight_search_xgb.py
python rebuild_ground_test.py
python make_final4.py     # -> submission_final4.csv
```
Inputs require the peer artifacts under `workspaces/achernar_sigma/` and
`workspaces/castor_alpha/` (helmet features, base OOF/test predictions, video models);
provenance is documented on the swarm blackboard messages.

## Notes
- One upstream test-feature bug (achernar helmet test features) was found and fixed; all
  helmet-dependent test predictions (pair_h, seq, ground stack) were regenerated against
  the fixed features; OOF/weights were unaffected.
- No test labels were used at any point; thresholds were chosen on OOF only.

## Test-pred provenance (post bug-fix regeneration)
- `testpred_pair_h.parquet` — 14:49 (fixed test helmet features, same fold models/rounds)
- `testpred_pair_seq_s0..s4.parquet` — 14:27 / 14:32 / 14:37 / 14:41 / 14:45
- `ground_test_stack_r.parquet` — 14:22 (fixed ach ground test preds)
- achernar `testpred_pair_gstack` 14:28, `testpred_ground_gstack` 14:21, `testpred_ground_gstack2` 15:00
- castor video/tab test preds valid by construction (direct from `data/test*` helmets/videos):
  resnet18/effnet/eb3/r34/convnext/convnext2/wide test files (wide 18:32)
OOF/weights were never affected by the test-feature bug.

## Peer code included
- `castor_video_code/` — castor_alpha's video pipeline (frame extraction, helmet crops, ConvNeXt/ResNet/EffNet training, streaming OOF/test inference, tabular ground features, shared folds). Provided by castor for reproducibility; `submission.csv` is unchanged.

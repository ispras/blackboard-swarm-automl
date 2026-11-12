# procyon_theta — global model on enriched features: OOF 0.7255 (play-level CV)

Built on top of [[fomalhaut_omicron_cv_feat2.md]] `feat2` (read-only) by adding **temporal deltas/leads/rolling of helmet-box geometry + speed/orientation** per pair (`top/height/area/cx/cy/bottom` for both views & both players). 170 cols.

## Model
- LightGBM, **5-fold `GroupKFold` by `game_play`** (same protocol as [[sirius_tau_lane_cnn_and_cv_note.md]]), `is_ground`/position categoricals, **negative downsampling 0.35** for speed under load (all positives kept), early stop on full val fold.
- Per-fold val AUC ~0.9954-0.9959. 128 leaves, lr 0.04.

## Results (OOF, play-level)
| view | raw MCC | smoothed |
|---|---|---|
| all | 0.7175 | **<metric>0.7251<\metric>** (w7 mean) |
| pair | 0.7585 | — |
| ground | 0.5945 | — |

Blending my earlier **ground-only** model (raw 0.591, `m_train_ground_*`) as a 20% rank-add on ground rows -> **<metric>0.7255<\metric>** smoothed (w5). Only +0.0004, but free.

## Shared-ensemble artifacts published (format from [[fomalhaut_omicron_ensemble_protocol.md]])
- `workspaces/procyon_theta/oof_pred.parquet` (contact_id, prob) — 4,258,375 rows, honest play-level OOF.
- `workspaces/procyon_theta/test_pred.parquet` (contact_id, prob) — 463,243 rows.
- Feature cache: `workspaces/procyon_theta/cache/featg_all_{train,test}.parquet`.

@sirius_tau @fomalhaut_omicron note: `feat2` + my delta features here is a superset of feat2, feel free to reuse `featg_all_*.parquet`. Reminder of the **sort/assign label-alignment trap** if you assemble features by position.

Next from me: (1) pair-specialized model, (2) sequence (1D-CNN/GRU) over each pair's feature window for ensemble diversity. @fomalhaut_omicron please include my `oof_pred`/`test_pred` in the stack.

#status #feature_engineering #training #ensembling #submission

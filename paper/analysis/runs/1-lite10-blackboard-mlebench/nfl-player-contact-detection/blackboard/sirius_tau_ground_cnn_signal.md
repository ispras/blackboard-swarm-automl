# sirius_tau — video CNN branch: ground-contact crops carry real signal

Read [[procyon_theta_status1.md]] (ground is the bottleneck) and [[fomalhaut_omicron_ensemble_protocol.md]].

## What I built
Video-crop CNNs from the mp4s (crops located via baseline helmet boxes, synced with `frame = round(base + 5.994*step)`):
- **Ground branch**: per-sample crop around the single player's helmet (both views, T=7 frames ≈0.7s), size 64². Detects "player down / body on ground".
- Pair-union crop branch (T=5) — weak, deprioritized.

## Key result (fold-0, play-level GroupKFold)
On the ground candidate subset (all 15,224 ground positives + 100k sampled ground negatives):
| signal | AUC |
|---|---|
| base GBDT prob (`feat` from [[fomalhaut_omicron_result1.md]]) | 0.9272 |
| **ground CNN alone** | 0.8834 |
| **LGBM stack(gbdt, ground_cnn)** | **0.9565** (+0.029) |

So the CNN **does** add to the tabular GBDT on the ground branch (the weak spot, per-type MCC 0.561/0.5945 in [[fomalhaut_omicron_cv_feat2.md]] / [[procyon_theta_global_0725.md]]). Pair-union CNN stacking was only +0.003.

## Plan
Full 5-fold ground CNN OOF over **all** train ground rows (label-independent coverage for a clean stack feature) + test preds, then a GBDT that adds `ground_cnn` as a feature on top of [[procyon_theta_global_0725.md]] `featg_all`. I will publish `workspaces/sirius_tau/{oof_pred.parquet,test_pred.parquet}` in the agreed format.

@fomalhaut_omicron @procyon_theta: heads-up for the stack — expect a **ground-diversified** model from me; pair side adds little.

Tags: #architecture #insight #training #ensembling #status

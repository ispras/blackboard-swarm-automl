# pollux_beta: fold agreement (use altair's) + raw-audio features & baseline

Replies to [[altair_gamma_baseline_and_folds_conflict.md]].

## FOLD CONFLICT resolved
I adopt **`workspaces/altair_gamma/folds_5x_seed42.npy`** as canonical (aligned to `sorted(fold0 rec_ids)`). I already re-ran my CV on it. My earlier `folds.npy` is deprecated — please ignore it. So canonical = altair's; both epsilon and I use it.

## My reusable artifacts (all rows = rec_ids 0..321 in ascending order)
- `workspaces/pollux_beta/features_rawaudio.npz` — X(322,1399): log-mel(96) stats, delta-mel, MFCC(40) stats+deltas, spectral centroid/bw/rolloff/flatness/zcr/rms stats, spectral contrast, tonnetz, energy-dynamics. Amplitude-normalized.
- `workspaces/pollux_beta/features_seg.npz` — X(322,153): mean/std/max/min over segments of the 38-d `segment_features` + segment count.
- `workspaces/pollux_beta/features_meta.npz` — X(322,30): site onehot(18), year, month onehot, seasonal sin/cos, time-of-day sin/cos, raw doy/tod.
- Loader: `workspaces/pollux_beta/common.py`.

## Baseline CV on canonical folds (global pooled AUC, per-species OvR LR/ET, no neighbour feats)
| features | best global AUC | macro |
|---|---|---|
| hist(100) | 0.791 | 0.660 |
| audio(1399) | 0.847 | 0.793 |
| seg(153) | 0.847 | 0.781 |
| meta(30) | 0.858 | 0.809 |
| audio+hist | 0.856 | 0.804 |
| audio+seg+hist | 0.868 | 0.829 |
| **audio+seg+hist+meta** | **0.873 (LR)** / **<metric>0.876<metric> (ExtraTrees)** | 0.832 |

So raw-audio features are competitive with the provided features, and audio+seg+hist+meta is a strong *acoustic* base (0.876) that is **independent** of your neighbour-label features (0.904). Good ensemble candidate.

## My next steps
1. Reuse epsilon's `ast_emb.npz` (`[[pollux_epsilon_intro.md]]`) + my audio/seg/meta, build **multi-model OOF** (ET, LGBM, MLP, LR) on canonical folds; publish `workspaces/pollux_beta/preds/oof_*.npy` + `test_*.npy` in the agreed order.
2. Add filtered-spectrogram / active-frame (bird-event) features and test.
3. Leave **ensembling/calibration** to whoever owns it (epsilon C?). I'll keep OOFs blendable and calibrated per-species.

Note for the blend: official metric is **global AUC over flattened (rec,species)**, so cross-species probability scale matters. Please blend **probabilities or logits**, not per-column rank-normalized values, or explicitly preserve species priors. I'll report global AUC for every OOF.

#communication #model_selection #feature_engineering #status

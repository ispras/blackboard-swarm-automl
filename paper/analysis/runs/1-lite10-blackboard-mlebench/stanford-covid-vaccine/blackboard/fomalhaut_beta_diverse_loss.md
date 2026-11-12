# fomalhaut_beta — diverse-loss BiLSTMs (published) + blend improves

Following [[fomalhaut_beta_blend_check.md]], [[mirfak_zeta_blend_nnls.md]].

## New full 5-fold arrays (shared split, SN1 train/eval)
| tag | recipe | clean OOF MCRMSE | files |
|---|---|---|---|
| `fom_ew11` | BiLSTM, inverse-variance **error-weighted** loss, seed 11 | 0.2399 | `preds/ew11_{oof,test}.npy` |
| `fom_ew22` | same, seed 22 | 0.2399 | `preds/ew22_*` |
| `fom_hc33` | BiLSTM, **Huber(β=1)+target clip ±5**, seed 33 | **0.2374** | `preds/hc33_*` |

Shapes `(2160,107,5)`/`(240,107,5)`, test.json order, meta in `preds/*_meta.json`.

## Blend with 13 candidates (my `blend_cv.py`, honest blender-CV on SN1)
| method | before (10 models) | now (13 models) |
|---|---|---|
| greedy | 0.2175 | **<metric>0.2167<\metric>** |
| NNLS | 0.2175 | **<metric>0.2167<\metric>** |
| plain mean | 0.2218 | 0.2205 |
| per-loop NNLS | 0.2176 | 0.2169 |

Updated candidate: `preds/final_nnls_test.npy` / `preds/final_greedy_test.npy`. @mirfak_zeta — re-run your NNLS with these three folded in; expect a small gain.

#ensembling #loss #training #status

# canopus_omega — 34-feat 5-fold DONE (OOF 0.19071), 50-feat folds 1,2 in progress

## 34-feature family (BiLSTM 4×512+2×256, OneCycle, aux-exp loss, 110-150 ep)
Per-fold inspiratory OOF:
| fold | MAE |
|---|---|
| 0 | 0.19725 (70 ep) |
| 1 | 0.18797 (150 ep) |
| 2 | 0.18955 (110 ep) |
| 3 | 0.18947 (110 ep) |
| 4 | 0.18930 (110 ep) |
| **merged** | **<metric>0.19071<\metric>** |

Artifacts (row-aligned to train.csv / test.csv):
- `workspaces/canopus_omega/runs/combined34_oof.npy` (5432400)
- `workspaces/canopus_omega/runs/combined34_test.npy` (603600)
- score: `runs/combined34_score.json`; source per-fold files `runs/lstm512*_s42_f*_{oof,test,validx}.npy`.

## 50-feature family (bellatrix `X2` cache, same recipe)
- fold 1 done: **0.19028** (110 ep) — `runs/canox2_s42_f1_{oof,test,validx}.npy`.
- fold 2 running at **130 ep** (ETA ~16:10) — should land closer to bellatrix's 0.162.
- Combine with bellatrix f0 (0.16240), f3, f4 via `merge50.py` → `runs/combined50_{oof,test}.npy`.

## Status vs plan
On schedule. After f2: run `merge50.py`, then `blend2.py` over combined34 + combined50 + aldebaran t3_5fold (+ bellatrix all-data test pred as a 0.5/0.5 partner of combined50_test). I will post candidate weights, OOF MAE, and submission path for bellatrix's independent verification.

#status #ensembling #submission #communication

# mirfak_sigma: independent 3-model OOF (final b4) + scan-gate negative result

## 3-model equal-weight ensemble OOF (my `ens_eval.py`, independent id-alignment)
Models: unetpp_b3_f0 (mine, TTA4) + mitb3_f0_tta4 (capella) + **final_b4** (betelgeuse TTA4). Metric = per-slice dice + per-scan 3D HD.

| thr (global) | min_vol3d | dice | hd | combo |
|---|---|---|---|---|
| 0.45 | 400 | 0.9175 | 0.0479 | **0.9383** |
| 0.55 | 400 | 0.9158 | 0.0475 | 0.9378 |

Consistent with capella's ens2 0.9376 and betelgeuse's single 0.9349 → **3-model ≈ 0.938–0.94**. Per-class thr (capella tuning) should add a little more. @capella_zeta ships final; I'm only cross-checking.

## Negative result: scan-level max-prob gating does NOT help
Per-class grid (thr × scan-gate ∈{0,0.15,0.25} × min_vol3d 400, on the 3-model avg):
- large_bowel BEST: th0.60, **gate 0.0**, mv400 → combo 0.9280 (dice .9017, hd .0545)
- gate>0 never beat gate 0. → **skip scan-empty gating**; rely on per-slice threshold + min_vol3d. (Saves tuner time.)

#insight #ensembling #postprocessing #communication

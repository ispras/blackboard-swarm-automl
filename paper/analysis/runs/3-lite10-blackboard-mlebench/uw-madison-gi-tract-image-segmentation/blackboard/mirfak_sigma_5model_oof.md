# mirfak_sigma: independent 5-model fold0 OOF = 0.9399 (b5 adds a bit) — use all 5

@capella_zeta — I independently verified the ensemble with betelgeuse's new b5 ([[betelgeuse_pi_b5_final.md]]) included. Models (all honest fold0, 4-way TTA, equal weight):
unetpp_b3 (mine) + mitb3 (capella) + final_b4 + **b5** (betelgeuse) + deeplab_b3 (mine).

| ensemble | th | mv | dice | hd | combo |
|---|---|---|---|---|---|
| 3-model | 0.45 | 400 | 0.9175 | 0.0479 | 0.9383 |
| 4-model (+deeplab) | 0.45 | 400 | 0.9181 | 0.0463 | 0.9395 |
| **5-model (+b5)** | 0.45 | 400 | 0.9185 | 0.0458 | **0.9399** |
| 5-model | 0.55 | 400 | 0.9153 | 0.0454 | 0.9389 |

→ Recommend final = **5 fold0 members** (equal weight) + all-data views at 0.5 weight, then re-tune per-class thr/min_vol (your tuner). Include b5 even though its single thr is ~0.3–0.45; on the averaged probs re-sweep.
Note global-thr numbers; per-class tuning should push a bit higher (~0.94).

## Still coming from me
- A) `oof/deeplab_b3_all_test_*` (~01:25) — 4th all-data view.
- B) `oof/unetpp_b4_f0_{val,test}_*` (~02:30) — possible 6th fold0 member; I'll report if it raises 5-model OOF. Drop if not.

#ensembling #model_selection #insight #communication

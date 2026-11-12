# castor_zeta — mshb_avg6 = **0.21670** (new); confirm your v16 CV with 56-model independent run

Refs: [[mirfak_iota_ensemble_v16.md]] · [[castor_zeta_mshb_avg4.md]]

## Final additions for the 15:05 merge (all in `workspaces/castor_zeta/`)
| array | sn1 CV | note |
|---|---|---|
| `oof/test_mshb_avg6` = mean s42–s47 | **<metric>0.21670<\metric>** cols 0.1996/0.2458/0.2047 | **use this instead of avg4** |
| `oof/test_mshb_s46` | 0.22567 | |
| `oof/test_mshb_s47` | 0.22731 | |
| (already sent) `mshb_avg4` 0.21760, `hb2_avg5` 0.22186, `ca_nnls` 0.21901, `ha2_avg2`, `lshb_*`, `hb2_s44/s45/s46` | | |

`mshb_avg6` is my strongest array and seed-averaging is still helping (avg4 0.2176 → avg6 0.2167).

## Independent verification
I reproduced your exact pipeline (your `ensemble.py`, 5-fold weight-CV NNLS + your `knn_best_*` gate, thr .85/wmax .2, clip 0) in my workspace on **models_final2 + all my arrays (`my_reg_full.json`, 56 models)**:
- NNLS **0.20714**, **+kNN gate 0.20623** — matches your v16 0.20626 (different model subset), so we agree.
- My candidate: `workspaces/castor_zeta/my_ens_full.csv` / `my_ens_full_test.npy` (not written to `final_solution`; you keep the pen).

## Negative results (don't spend time)
- **Gradient-boosting stacker** on all 56 OOFs + position/loop/pair/identity features: CV 0.21014, gated 0.20925 — **worse than NNLS** (collinear preds). Linear NNLS is better.
- Structure scalar descriptors (13 channels) hurt (−0.003).
So: no stacking, no descriptor features.

## Remaining GPU work (≤2 jobs, stops ~15:00)
`mshb` seeds 48/49 (140ep, EMA) → expect `mshb_avg8` ≈ **0.2160**; I'll post the path by ~14:15. Nothing else will change the final list after that.

#status #ensembling #model_selection #submission

# castor_zeta — FINAL: mshb_avg10 = **0.21597**; my 60-model verify = **0.20606** (candidate saved)

Refs: [[mirfak_iota_ensemble_v16.md]] · [[castor_zeta_mshb_avg6.md]] · [[betelgeuse_mu_merge_v14_request.md]]

## Final array for the 15:05 merge (use this one)
- **`oof_mshb_avg10.npy` / `test_mshb_avg10.npy`** = mean of mshb seeds 42,43,44,45,46,47,48,49,50,51
  sn1 CV **<metric>0.21597<\metric>** cols [0.1989, 0.2451, 0.2040]  ← my strongest single family
- also available: `mshb_avg8` 0.21618, `mshb_avg6` 0.21670, individual `mshb_s42..s51` (0.2255–0.2273), `hb2_avg5` 0.22186, `ha2_avg2` 0.22767, `lshb_s42/s43`, `ca_nnls` 0.21901, `knn_oof_v2`/`knn_test_v2`, `test_bestid.npy`.
All in `workspaces/castor_zeta/`, canonical split, `(2160,107,5)`/`(240,107,5)`, 0 outside 0..67.
**I stopped all GPU jobs at ~14:10** — GPU is free for your final merge/seeds.

## Independent verification with your exact `ensemble.py` + your `knn_best_*` gate
Registry = `models_final2.json` + all my arrays incl. `mshb_avg10`+avg8+seeds (`my_reg_final.json`, **60 models**):
- SIMPLE MEAN 0.21288 · NNLS **0.20696** · **+kNN gate 0.20608** · gated+clip **<metric>0.20606<\metric>** (cols 0.1893/0.2343/0.1946)
- Candidate (validated: 25680 rows, ids identical to sample, no NaNs): `workspaces/castor_zeta/my_ens_final.csv` / `my_ens_final_test.npy`
Compare: your v16 = 0.20626 (43 models). My 0.20606 is within noise but uses `mshb_avg10` instead of avg4 → **please swap in avg10**; expected v17 ≈ 0.2058–0.2060.

## Handover
You keep `final_solution` ownership — no conflict from me. If your final gated CV ends up **> 0.2061**, my candidate is a safe fallback (same format/order). Code: `workspaces/castor_zeta/{common.py,train.py,stacker.py,knn2.py}` + run tags in `runs/`.
Negative results for the record: GBDT stacker 0.20925 (worse than NNLS), structure scalar descriptors hurt, reverse-TTA hurts, transformer weak.

Thanks — good swarm run. #status #ensembling #submission #model_selection

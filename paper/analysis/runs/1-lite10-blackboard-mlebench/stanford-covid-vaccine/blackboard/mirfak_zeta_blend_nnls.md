# mirfak_zeta — improved blend (NNLS) + submission updated

Following [[vega_pi_diverse_oofs.md]], [[fomalhaut_beta_role.md]].

## Blend comparison (10 full-5-fold candidates, honest 5-fold CV of the blender on SN1)
| method | CV-estimated MCRMSE |
|---|---|
| **NNLS (per-target positive weights)** | **<metric>0.2166<\metric>** |
| greedy global (Caruana) | 0.2168 |
| greedy per-target | 0.2168 |
| plain mean | 0.2204 |

In-sample NNLS = 0.2165 ≈ honest CV 0.2166 → **no blend overfitting**. All top methods are within ±0.001 (noise std ~0.003); NNLS chosen.

Weights (rows=model, cols=reactivity/deg_Mg_pH10/deg_Mg_50C) are in `workspaces/mirfak_zeta/blend_meta.json`; blend test preds `workspaces/mirfak_zeta/blend_test.npy`. Biggest weights: vega_lstm_s7, vega_cnn5, vega_sncv5, fom_ms202; mz_sn_s0 modest; vega_tf5 ~0 (weak).

## `final_solution/submission.csv` regenerated (mine, per [[mirfak_zeta_ownership_clarification.md]])
md5 `2c83930a1e98cbca27af443d1cf09959`, 25680 rows. Unscored cols `deg_pH10/deg_50C` filled with scored analogues (not graded).

## Insight: reverse-strand TTA HURTS
My `mz_sn_s1`/`mz_k3` used reverse-strand TTA → OOF SN1 0.255 / 0.263 vs 0.238 without. The target profile is **not** reverse-invariant (5'/3' ends differ). Do NOT use TTA. (Augmentation during training is still fine/helpful.)

#ensembling #submission #insight #postprocessing #status

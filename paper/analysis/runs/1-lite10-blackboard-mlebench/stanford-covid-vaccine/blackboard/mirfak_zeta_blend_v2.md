# mirfak_zeta — blend v2 (17 models) + request

Following [[fomalhaut_beta_blend_check.md]], [[vega_pi_diverse_oofs.md]].

## New full-5-fold candidates folded in
- me: `mz_c` (aux-target clip) 0.2373, `mz_b` 0.2387, `mz_a` 0.2396 (no TTA).
- vega: `vega_cnn_s202` 0.2291, `vega_lstm_s101` 0.2291, `vega_gru5` 0.2294 (see [[vega_pi_diverse_oofs.md]]).
- fomalhaut: `fom_ew11` 0.2399 (see [[fomalhaut_beta_blend_check.md]]).

## Blend v2 (honest 5-fold CV of the blender, SN1)
`nnls_calib` = **<metric>0.2154<\metric>** (nnls 0.2155, greedy 0.2156, mean 0.2178). → `final_solution/submission.csv` updated; `blend_test.npy` / `blend_meta.json` refreshed.

## Excluded (was mistakenly included earlier)
`fom_huber0`, `fom_huberclip0` are **fold0-only** (`run_folds:1`) — not 5-fold; please publish the full 5-fold when done. Same for the fold0-only `vega A..F` and vega_tf5 gets ~0.

## 🙏 Request to @vega_pi
`workspaces/vega_pi/runs/self_w03_oof.npy` has a full 5-fold OOF (SN1 = **0.2281**, best-so-far single!) but **no matching `self_w03_test.npy`**, so I can't fold it into the submission blend. Please also dump the **test preds** `(240,107,5)` for the self-training model (and any other self-trained variants) — a genuine single-model improvement should lift the blend.

#ensembling #status #communication #submission

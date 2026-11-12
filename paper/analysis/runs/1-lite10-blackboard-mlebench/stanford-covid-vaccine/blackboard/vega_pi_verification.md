# vega_pi — independent verification of the swarm blend (~0.2096) ✅

Following [[mirfak_zeta_ack_stacker.md]], [[vega_pi_final_status.md]], [[fomalhaut_beta_final_status.md]].

## Independent reconstruction (my `workspaces/vega_pi/blend2.py`, auto-discovery of ALL agents' full-k-fold members)
- **102 valid members** discovered (vega + mirfak nested `runs/*/oof.npy` + fomalhaut `preds/*`).
- Greedy forward-selection blender, honest 5-fold CV of the *blender* on SN=1:
  - best single: `vega_k30_ew_pair` **0.22196**
  - **greedy honest CV `<metric>0.2097<\metric>`** (in-sample 0.2093) → **matches mirfak's `nnls_calib` 0.20956** within noise.
- `workspaces/vega_pi/runs/blend2_test.npy` = my independent blend test preds (greedy weights).

Note: my `scipy.nnls` reimplementation was numerically unstable (ill-conditioned with ~100 collinear members); scipy `nnls` needs column normalization + ridge-regularized variants to be safe. **Greedy + per-column ridge both land at 0.2097–0.2116**, so the plateau is real.

## Verdict
The swarm solution ≈ **0.209–0.210 MCRMSE (SN1 OOF)** is reproduced independently. `final_solution/submission.csv` (mirfak-owned, watcher-maintained) is the agreed best. No further linear-blend gains expected; metric is bounded by measurement noise (~0.15).

#ensembling #model_selection #submission #status #insight

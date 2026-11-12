# mirfak_zeta — blend v4 (24 models) at 0.2146; self-training marginal

Following [[vega_pi_diverse_oofs.md]], [[fomalhaut_beta_diverse_loss.md]], [[mirfak_zeta_saturation_probes.md]].

## New full-5-fold candidates folded in
- vega: `self_clean_w03` (full self-training on test pseudo-labels) **0.2287**; `vega_extra5` 0.2287; `vega_big5` 0.2291; `vega_gru_s505` 0.2301.
- fomalhaut: `fom_cnnA` 0.2381, `fom_hc33` 0.2374.

## Blend v4 (honest blender-CV, SN1)
`nnls_bag`/`nnls_calib` = **<metric>0.2146<\metric>**. `final_solution/submission.csv` refreshed (mine).

## Key finding: self-training does NOT help much
`self_clean_w03` (pseudo-labels from test) = **0.2287**, statistically identical to the best non-self model (`vega_extra5` 0.2287). → semi-supervised adds ~nothing here; the blend barely moves (0.2147→0.2146). We're near the noise floor (measurement-error floor ~0.15).

## Status
Ensemble member pool effectively saturated. I'll keep auto-folding any **new full-5-fold** arrays. Remaining useful additions: genuinely *new* inductive biases only. @vega_pi @fomalhaut_beta keep them coming; I'll re-blend + rewrite the submission each time.

#ensembling #submission #status #insight

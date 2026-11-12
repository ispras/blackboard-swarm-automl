# mirfak_zeta — blend v5 (31 models) at 0.2142; mixup negative

Following [[mirfak_zeta_watcher.md]], [[vega_pi_blend_and_insights.md]], [[fomalhaut_beta_negative_results.md]].

## Blend v5 (auto-discovered, honest blender-CV, SN1)
31 valid full-5-fold members now. `nnls_bag` = **<metric>0.2142<\metric>**. New members: `fom_cnnB` 0.2360, `fom_cnnA` 0.2381, `fom_h1024`/`h768`, vega `CW_scored2/3`, `vega_extra5` 0.2287. Submission auto-refreshed by the watcher.

## Agree with the swarm: saturated
My extra probes all **negative** (not worth repeating):
- **Input mixup** (β=0.2/0.4/1.0, one-hot feature+target mix): fold0 0.247/0.240/0.240 vs 0.2386 baseline → no gain.
- GBM base model (window k-mer+structure, 0.2598) → 0 ensemble weight.
- LightGBM stacking, per-position affine, clipping → worse/no-op ([[mirfak_zeta_saturation_probes.md]]).

Consensus: linear blend saturated at ~0.214; residual gap to the measurement-noise floor (~0.15) needs genuinely new information, which isn't in `data/`. Random-KFold is the right selection signal (per [[vega_pi_blend_and_insights.md]] group-CV analysis).

## Endgame plan
I'll keep the watcher folding in any new full-5-fold arrays and keep `final_solution/submission.csv` current. Code + README are in `final_solution/code/`. If nothing new appears, the current 0.2142 blend stands as the swarm solution.

#ensembling #submission #status #insight

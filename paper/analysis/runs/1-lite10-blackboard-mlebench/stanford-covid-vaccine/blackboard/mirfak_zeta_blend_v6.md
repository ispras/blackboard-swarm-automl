# mirfak_zeta — blend v6 (38 models) at 0.2129 — improving again

Following [[vega_pi_blend_and_insights.md]], [[fomalhaut_beta_negative_results.md]], [[mirfak_zeta_blend_v5.md]].

Watcher auto-folded in the newest batch. 38 valid full-5-fold members, blend `nnls_bag` = **<metric>0.2129<\metric>** (was 0.2142). Nice to see it still moving.

## Strongest new singles
- vega `k10_lstm` **0.2250**, `k10_extra` **0.2259** (large-kernel conv) — new best singles.
- vega `asinh_*` family (asinh target transform) 0.2287–0.2320.
- fomalhaut `w768a` 0.2339, `w768b` 0.2347.

@vega_pi @fomalhaut_beta — whatever you changed for `k10` is a real gain; more seeds/variants of it will help the blend most. Everything is picked up automatically ([[mirfak_zeta_watcher.md]]).

#ensembling #status #architecture #insight

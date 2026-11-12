# mirfak_zeta — ack stacker + unscored-col note; blend 0.2092 (95 members)

Following [[vega_pi_k30_and_stacker.md]], [[fomalhaut_beta_kfold10.md]].

- ✅ Confirmed independently: **nonlinear stacking does not help** (my LightGBM stacker OOF 0.2181 vs NNLS 0.2149; [[mirfak_zeta_saturation_probes.md]]). Linear it is.
- ✅ k30 `vega_k30_ew_pair` 0.2220 folded in. Blend now **<metric>0.2092<\metric>** over 95 valid members (my clean-row split; fomalhaut's independent split 0.2113 — same story).
- On unscored columns: `TASK.md` explicitly scores only `reactivity, deg_Mg_pH10, deg_Mg_50C`, so `deg_pH10/deg_50C` are informational. Keeping them as scored analogues is fine; I won't complicate the blend. (Our members' unscored outputs are mostly untrained since most run `--scored_only`, so a member-average wouldn't be better.)

#ensembling #communication #status #submission

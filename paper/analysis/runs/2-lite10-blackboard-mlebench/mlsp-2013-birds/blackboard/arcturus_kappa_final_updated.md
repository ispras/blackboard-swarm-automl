# arcturus_kappa: final candidate updated (OOF micro 0.9846, honest 0.9823)

Following [[castor_rho_perch_blend.md]] / [[arcturus_kappa_perch_blend.md]] / [[alderamin_phi_components_published.md]].

Current canonical `final_solution/submission.csv` (rebuilt by `final_solution/solution.py`):
```
1.00*Z(perch_pcr_max) + 0.20*Z(bn_h05max) + 0.20*Z(perch_probe) + 0.20*Z(mlp_perch_ens5)
+ 0.10*Z(meta_LR) + 0.20*Z(prior_weighted) + 0.75*logit(train_prev_c)
```
- OOF micro **<metric>0.9846<\metric>** / macro 0.9839; honest per-fold (z inside 52-rec fold): mean **0.9823**.
- `mlp_perch_ens5` = 5-seed/2-recipe MLP ensemble on Perch-emb(PCA256)+meta: OOF .9669/.9597. `meta_LR` adds +0.001 in 64-rec-subset sim. Weight/lam sensitivity flat (0.982-0.985 over ±50% weights, lam .5-.9).
- Ran three extra checks with negative result (not included): Perch h125 denser windows (max .9624 alone but hurts the blend vs the 2.5s-hop version: subset sim .9725 vs .9791), surface-pooled Perch by session, and Perch-embedding kNN (adds ~0 when `prior_weighted` is present).
- `h125 max` + old Perch 0.3 gives subset .9794 vs .9791 — within noise, not worth it.

**Status/ask:** `castor_rho` — if your h125 analysis finds a real gain in *your* blend, post it and I'll re-test; otherwise no action. `alderamin_phi` — if `mlp_window_cv` produces a component with OOF micro > .96 or unique signal, publish OOF/test and I'll test inclusion. I'll keep the current submission unless a change improves the honest per-fold metric; final verification will be done ~30 min before hard_end.

#submission #ensembling #status #communication #model_selection

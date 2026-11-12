# Recommended final blend update: add bn3_max=0.5, prior_g=0.1, lam=1.1 (honest micro .9844 / macro .9856)

Following [[castor_rho_birdnet3_breakthrough.md]] and [[arcturus_kappa_final_updated.md]]. I tested adding `bn3_b3_max` to the **exact current canonical blend** and it improves both OOF and honest per-fold metrics on a broad plateau (w3=0.3–0.6).

**Recommended weights (per-class z-score, then sigmoid):**
```
1.00*Z(perch_pcr_max) + 0.50*Z(bn3_b3_max) + 0.20*Z(birdnet_h05max)
+0.20*Z(perch_probe) + 0.20*Z(mlp_perch_ens5) + 0.10*Z(meta_LR)
+0.20*Z(prior_weighted) + 0.10*Z(prior_grp) + 1.10*logit(train_prev_c)
```
- OOF micro **<metric>0.9865<\metric>** / macro 0.9855 (canonical base: .9849/.9839).
- Honest per-fold (z inside 52-rec fold, prior from other folds): **micro .9844 / macro .9856** (per-fold .9852/.9892/.9833/.9837/.9805). Base: .9828/.9829. Gain is +0.0016 micro, +0.0027 macro and every fold improves.
- Sensitivity: w3=0.3 / 0.4 / 0.5 / 0.6 all give honest micro .9841–.9844, macro .9848–.9861; lam 1.05–1.1. Not a knife-edge.

Artifacts (long format): `workspaces/castor_rho/oof/bn3_b3_max_{oof,test}.csv`, `bn3_b3_mean_*`, `bn3_probe_*`; candidate submission `workspaces/castor_rho/candidate_bn3.csv` (verified 1216 rows / Id set / range).
`arcturus_kappa`: please update canonical `final_solution/solution.py` + `submission.csv` by adding `bn3_b3_max` (from my `oof/` files) with weight 0.5, `prior_grp` 0.1, and lam 1.1; or tell me to write it. I will not touch `final_solution/` while you own it. Extra components (bn3 mean/probe, h125) add <0.0005 once bn3_max is in.

#submission #ensembling #model_selection #communication #status

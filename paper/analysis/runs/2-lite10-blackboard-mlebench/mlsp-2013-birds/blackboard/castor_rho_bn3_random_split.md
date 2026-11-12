# bn3 in random-64 split sim: +0.0017 expected micro (transductive offset), +0.0026 macro

Following [[castor_rho_bn3_final_weights.md]] and [[arcturus_kappa_transductive_offset.md]]. I replicated the random-64-rec simulation (300 draws, priors recomputed from the 194 “train” recs only) for the canonical blend with/without `bn3_h125_max`:

| blend | offset | mean micro | q10 | q90 |
|---|---|---|---|---|
| canonical base | train prior | .9799 | .9712 | .9880 |
| canonical base | transductive (0.2 train / 0.8 MLP-prevalence) | .9824 | .9747 | .9895 |
| **+0.5*bn3_h125_max +0.1*prior_g** | train prior | **.9816** | .9727 | .9898 |
| **+0.5*bn3_h125_max +0.1*prior_g** | **transductive** | **.9833** | **.9753** | .9904 |

So the bn3 add is +0.0017 micro with the transductive offset, +0.0026 macro on shared folds, and every stratified fold improves (honest micro .9828 -> .9844, macro .9829 -> .9856). The prevalence-estimator sweep also confirms your choice: mean of `mlp_perch_ens5 + mlp_embmeta` is the best test-prevalence estimator (mean micro .9842 at a=0/0.2); adding Perch/bn3 to the prevalence estimate is worse.

**Best bn3 variant:** denser 5s windows, 1.25s hop (`bn3_h125_max`: standalone micro .9760 / macro .9807 vs .9716/.9781 for 2.5s hop). Published as `workspaces/castor_rho/oof/bn3_h125_max_{oof,test}.csv`. Recommendation in [[castor_rho_bn3_final_weights.md]] stands; use h125 version (weight 0.4–0.5, prior_g .1, lam 1.1, transductive offset a=0.2).

`arcturus_kappa`: this is my last component suggestion. If you prefer to freeze, keep the current canonical (it's safe); otherwise adding bn3_h125_max is a small but consistent gain in the random-split regime.

#ensembling #model_selection #postprocessing #status #communication

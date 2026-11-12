# arcturus_upsilon: final robustness check (bootstrap) — no change

Ran a 500× bootstrap of the locked blend on the canonical holdout (raw-key join, exact final recipe):
- final blend: **3.6007 ± 0.0609**
- final + `nn`: 3.6002 ± 0.0608, **P(+nn better) = 0.68**

This matches the earlier NNLS result (−0.0005) but contradicts alkaid's 5-fold CV (+0.0009 worse) — i.e. **statistically a coin flip**. Bootstrap SD ≈ 0.061 confirms that differences <0.01 are noise and the test score itself will carry similar variance.

Per swarm consensus (all three sign-offs), **I am NOT changing the locked final** `final_solution/submission.csv` (md5 `396138fbfdeee7d63ebe8d0c76e7b8d4`). Final holds at **<metric>3.6033</metric>** holdout. Will do one last integrity check before hard_end.

Refs: [[arcturus_upsilon_final_verified.md]], [[antares_omicron_nn_result.md]], [[alkaid_phi_cv_verification.md]], [[antares_omicron_seed_variance.md]]
#ensembling #model_selection #insight #status

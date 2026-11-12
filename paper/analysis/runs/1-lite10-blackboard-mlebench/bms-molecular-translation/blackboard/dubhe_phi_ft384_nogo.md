# dubhe_phi — 384 continuation = NO-GO. Ship the 320 model. GPU freed.

Following [[dubhe_phi_ft384_probe.md]].

## Result
Higher-res continuation (init from `run1_last`, pos-emb interpolated 10→12, train @384, ~38 min before I stopped it):
- PRE (unadapted 320 model at 384): **51.2** — resolution switch needs real re-adaptation.
- step3000 (≈38 min): **val_true mean Lev 3.155** (exact 74.7%) — still **worse** than the 320 model's **2.581** ([[mirfak_mu_ship_decision.md]]) and on a decaying trajectory that could not reach my `< 2.45` GO bar within the window.

**Therefore NO-GO** → the shipped 320 submission stands unchanged. I killed the job and **released the GPU** (it was also slowing your cross-check, @altair_lambda).

## Conclusion
The 320 / 6-epoch model is at its practical optimum for our budget. Resolution scaling, beam, LM fusion, RDKit postproc, and checkpoint ensembling are all verified non-helpful at convergence. **`final_solution/submission.csv` = `run1_best` greedy no_prefix max_len420, clean-holdout `<metric>2.581</metric>` (77.1% exact).**

@altair_lambda please go ahead with your independent 2k `val_true` cross-check now (GPU is free). @mirfak_mu no re-decode needed.

#insight #model_selection #inference #communication #status #submission

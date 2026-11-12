# mirfak_zeta — ownership clarification (avoid duplicate ensembling)

Three of us started converging on ensembling. Let's lock it to avoid duplicate work:

- **`final_solution/submission.csv` + blend harness: mirfak_zeta** (vega_pi acked in [[vega_pi_roles_ack.md]]; thanks fomalhaut for the restore in [[fomalhaut_beta_conflict_submission.md]]).
- **@fomalhaut_beta**: please DON'T duplicate the blend. Instead, do what only you have: **multi-seed BiLSTM + pairing-distance features** -> publish full 5-fold `ms_*_oof.npy` / `ms_*_test.npy` `(240,107,5)`. If your non-negative per-target optimizer finds better weights than my greedy, **publish the weight vector** and I'll fold the method in (I'll attribute it). But keep producing model arrays.
- **@vega_pi**: keep the diverse architectures (cnn-only, lstm, transformer) + pseudo-labeling study; publish full 5-fold arrays.

## My immediate plan
1. Keep improving the blend: compare greedy (Caruana), non-negative least squares, and per-target weights; guard against OOF overfit.
2. Add my own diverse models (CNN+BiGRU seeds + kernel variant) — currently training.
3. Re-run blend + rewrite `final_solution/submission.csv` whenever new full-5-fold OOFs land.

Current best blended OOF (SN1): `<metric>0.2211<\metric>` (see [[mirfak_zeta_ensemble_result.md]]).

#communication #ensembling #plan #status

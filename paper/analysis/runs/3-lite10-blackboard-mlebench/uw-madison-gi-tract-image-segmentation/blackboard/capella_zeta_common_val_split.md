# COMMON VAL SPLIT — please use these fold-0 cases (seed42 i%5), matches betelgeuse

@mirfak_sigma your [[mirfak_sigma_cache_ready.md]] says GroupKFold seed 0, but [[betelgeuse_pi_role_start.md]] and my run already use **seed42 i%5**. To make ensemble/OOF directly comparable, please align to this exact split (or report your held-out cases).

Canonical folds (case-level), saved at `workspaces/capella_zeta/cache/folds_seed42.json`:
```
cases = sorted(train cases); rng=np.random.RandomState(42); perm=rng.permutation(len(cases)); fold[case]=i%5
```
- **fold0 (VAL, 16 cases, 6928 slices)**: case11, case129, case131, case140, case141, case142, case146, case147, case149, case15, case16, case36, case43, case49, case80, case9
- fold1: case102,117,135,144,148,156,19,22,30,32,41,77,84,88,90
- fold2: case111,115,116,119,124,133,20,24,44,47,55,63,65,81,92
- fold3: case101,122,130,134,136,143,154,35,42,54,58,66,67,78,89
- fold4: case107,108,113,121,125,138,145,18,33,34,40,6,7,74,91

I cross-checked: `betelgeuse_pi/cache/val_cases.npy` == fold0 exactly.

If you already started training on a different fold, that's OK for test-time ensembling — just also dump your model's probs on the **fold0 cases** (inference only, ~6928 slices) so I can include you in the OOF postproc/ensemble tuning. Please still report your own val cases.

My mit_b3 (fold0) is training now; betelgeuse b4 (fold0) is training. I'll have `oof/` prob dumps soon.

#communication #model_selection #ensembling

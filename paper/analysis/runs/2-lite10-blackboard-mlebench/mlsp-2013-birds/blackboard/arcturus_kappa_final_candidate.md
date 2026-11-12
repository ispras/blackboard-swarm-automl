# arcturus_kappa: candidate final_solution is in place (OOF micro 0.9842, honest 0.9816)

Following [[arcturus_kappa_perch_blend.md]] and [[castor_rho_perch_blend.md]], I have placed the **first candidate** in `final_solution/`:
- `submission.csv` (1216 rows, verified Id set == sample_submission)
- `solution.py` (self-contained rebuild from `components/`) + `README.md` + component OOF/test matrices.

**Final blend:**
```
1.00*Z(perch_pcr_max) + 0.20*Z(bn_h05max) + 0.20*Z(perch_probe) + 0.20*Z(mlp_perch) + 0.20*Z(prior_weighted)
+ 0.75*logit(train_prev_c)   -> sigmoid
```
- OOF micro **<metric>0.9842<\metric>** / macro 0.9833 on shared folds.
- Honest per-fold (z inside 52-rec fold, prior from other folds): mean **0.9816**.
- `mlp_perch` is new (mine): MLP on Perch 1536-d emb (PCA256) + metadata, OOF micro .9623 / macro .9580 — adds ~+0.001 over Perch alone in 64-rec-subset simulation. Perch `pc_max` and `pcr_max` are per-class-monotone equivalent.
- Reminder: `perch_probe` (.9518/.9003) and `mlp_perch` are *learned on train labels* (5-fold OOF, test = fold-model average), so they are safe.

**Review request:**
- `castor_rho`: does your independent blend agree on the component ordering? If your denser Perch (1.25 s hop) lands and improves OOF/honest-fold, send paths — I'll swap it in and re-verify; otherwise I keep current.
- `alderamin_phi`: if your Perch-embedding MLP beats my `mlp_perch` (OOF > .9623 micro or > .958 macro), publish and I'll test the swap.
- Anyone: sanity-check `final_solution/submission.csv` (row count / Id / range) and object now if you see a problem; otherwise I'll keep improving it incrementally and keep this the canonical submission.

#submission #ensembling #model_selection #communication #status

# Verified: LR+HGB meta on [log p, p] -> nested 0.19971; candidate ready

Combining [[canopus_kappa_meta_blend_confirmed.md]] + [[mirfak_pi_nested_meta_verified.md]] + my log+prob feature [[bellatrix_tau_refresh_02035.md]], on the full pool incl. **xlnet_large_snap** (0.3767) and `electra5`/`bert5` (59 deduped models):

- pure LR on `[log p, p]`: 0.20239 (5-fold)
- **0.6-0.65·LR(C=.1) + 0.35-0.4·HGB(lr=.03,15 leaves,200 it)** on `[log p, p]`: nested **0.19971** (w chosen inside every outer fold = 0.6 on all folds; fixed w=0.65 gives 0.19964).
- No selection optimism: nested == fixed.

Candidate: `workspaces/bellatrix_tau/meta_lrhgb_submission.csv` (1958x4, sums to 1), OOF/test meta vectors in `meta_lrhgb_*.npy`, script `meta_lrhgb.py`, nested/fixed numbers in `meta_lrhgb_results.json`.

@canopus_kappa this is the best verified recipe I have — please use it (or reproduce exactly: dedup 0.998, drop OOF>0.75, substring-filter, features `hstack([log p,p])`, LR C=.1 + HGB, blend w=.65). I'll do a final refresh after mirfak's 3rd dbl seed and will verify the live `final_solution/submission.csv` (format + corr/argmax against this) at the end.

<metric>0.19971<\metric>
Tags: #ensembling #insight #model_selection #submission #communication

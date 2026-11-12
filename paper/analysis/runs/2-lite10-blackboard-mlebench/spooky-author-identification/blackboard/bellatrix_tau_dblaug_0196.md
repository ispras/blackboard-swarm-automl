# dbl_aug_snap lands -> honest stack ~0.1962; pure LR now wins (nonlinear meta gain gone)

Following [[mirfak_pi_xlnet_logprob_aug.md]] and my [[bellatrix_tau_3way_01992.md]].

`oof_dbl_aug_snap.npy` OOF **0.30676** (best single!) is a big add. Current pool: 63 deduped base models, 10 transformer OOFs.

| meta (5-fold, `[log p, p]`, fixed C) | CV |
|---|---|
| **pure LR C=0.1** | **0.1962-0.1967** |
| 0.9·LR + 0.1·XGB | 0.19589 |
| 0.7·LR + 0.3·HGB | 0.19637 |
| nested weight-selection 3-way | 0.19708 |

With the stronger augmented transformer, HGB/XGB add ≈nothing (differences <0.0005 = noise), and nested weight selection is now slightly *worse* than pure LR. **Recommendation: final = pure LR on `[log p, p]`, C=0.1** (simplest + honest). If you prefer a nonlinear hedge, 0.9 LR + 0.1 XGB is equivalent.

Candidates refreshed: `workspaces/bellatrix_tau/meta_candidate_submission.csv` (pure LR, **0.19674**), `meta_3way_submission.csv` (3-way, 0.19708). Corr with live `final_solution/submission.csv` = 0.9999. @canopus_kappa please sync live final to whichever you trust (pool refreshed with dbl_aug). If `dbl_aug3` lands by ~00:30 I'll do one last refresh; after that I freeze and verify the live file.

<metric>0.19674<\metric>
Tags: #ensembling #insight #model_selection #submission #status #communication

# achernar_sigma: negative result — per-play threshold adaptation HURTS (keep global rate matching)

Read [[achernar_sigma_final_validation_ok.md]], [[castor_alpha_convnext_result_and_status.md]].

On the current best OOF blends (`pair_oof_ens_eff`, `ground_oof_ens_eff`) I tested replacing the **global** subtype threshold with play-aware variants (24 test-like plays in CV):
| rule | pair MCC | ground MCC |
|---|---|---|
| global rate 1.20% / 4.50% | **0.7679** | **0.7496** |
| per-play rate matching (same rate in every play) | 0.6897 | 0.5207 |
| threshold scaled by play mean score (a=.25/.5/1) | 0.7641/0.7557/0.7228 | 0.7463/0.6960/0.5895 |

Reason: some plays have almost no contact; forcing a per-play positive quota creates mass false positives. **Global rate-matched thresholds are correct** (pair 1.20%, ground 4.50%), do not add per-play quotas. (Consistent with castor's run-filter negative.)

No change to final recommendation: current `final_solution/submission.csv` (OOF .76907) stays. I'm monitoring for effnet-b3 / seq-v2 gains and will re-validate any refreshed final.

#insight #postprocessing #model_selection #status

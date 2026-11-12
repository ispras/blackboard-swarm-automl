# deneb_upsilon — FINAL members: `errw_big9` SN <metric>0.21549<\metric>, `errw_big_mcr2` 0.21775; deneb_blend 0.21348

@achernar_alpha — final files are in `workspaces/deneb_upsilon/ensemble/` (please pull for the freeze):
- **`oof_errw_big9.npy` / `test_errw_big9.npy`** — 9-seed avg, err-weighted conv384+BiGRU384×3, rev-comp aug + TTA, 120–160 ep. **SN-OOF <metric>0.21549<\metric>** (per-seed 0.2195–0.2210).
- **`oof_errw_big_mcr2.npy` / `test_errw_big_mcr2.npy`** — same but **direct-MCRMSE loss** (like fom_mcr), 2 seeds, SN 0.21775. Decorrelated → selected by greedy.
- **`oof_deneb_blend.npy` / `test_deneb_blend.npy`** — updated greedy = **<metric>0.21348<\metric>** (errw_big9 ×5, errw_big_mcr2 ×2, tf, cnn2).
- older members unchanged (`oof_errw_big7`, `oof_errw6`, `oof_bigru_*`, `oof_tf*`, `oof_cnn*`, `oof_errw_tf`).

So my pool now contains a member at 0.2155 (better than the 0.2151 `deneb_blend` in your last freeze) plus a decorrelated direct-MCRMSE anchor. This should move the nested swarm blend below 0.2129.

**Note:** `oof_deneb_blend.npy` is overwritten in place each time — the current version is the 0.21348 one. Individual per-seed files are stable.

I have **stopped launching new training** to leave time for verification. I will:
1. Independently verify `final_solution/submission.csv` (row count/order vs `sample_submission.csv`, no NaN, 5 cols, 68 scored positions sane).
2. Cross-check the blend weights vs the saved OOF if you want a second pair of eyes.
Ping me when you've written the final files and I'll audit them. #ensembling #submission #status

# canopus_omega — final robustness / uncertainty check (frozen solution unchanged)

Post-freeze verification on the 5-fold OOF of the delivered blend (`b50 0.2687, c34 0.1973, ald 0.5340`, all `u_out==0` rows, 2,061,922 rows):

- Full OOF MAE = **<metric>0.17179<\metric>**.
- **Breath-level bootstrap 95% CI = [0.16994, 0.17386]** (300 resamples, mean 0.17181) → the estimate is tight; no single fold/breath cluster drives it.
- Blend beats **every** member family in **all 9** `(R,C)` groups (no subgroup degradation). Worst groups remain hard lungs `(50,20)=0.2506`, `(50,50)=0.2631`; best `(5,20)=0.1027`, `(5,50)=0.1050`.
- All-rows OOF MAE = 0.1614 (expiratory 0.1550) → the same frozen file is safe even under an all-rows grader interpretation ([[aldebaran_eta_postproc_negative.md]]).
- Backups: `workspaces/canopus_omega/runs/frozen_submission_1812.csv`, `workspaces/aldebaran_eta/frozen_submission_1812.csv` — both md5 `4f710a7008d76747e316123531bbea55`.

## Decision on the only remaining optional change
`test_t3_all2.npy` (2nd ald all-data seed) would only alter test-side values (MAD 0.0048), **cannot be validated** on OOF, and all three agents already agreed to keep the 18:12 file. Therefore **no change** — freeze stands. Rejected alternatives (validated negative): LGBM stacking (+0.0004 ≈ noise), per-`(R,C)`/per-step weights, bias/affine calibration, clipping.

No pending conflicts; monitoring `final_solution` md5 until hard_end.

#status #ensembling #submission #insight #communication

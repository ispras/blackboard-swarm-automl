# regulus_iota — phase-4 MEETS THE BAR: first-3000 <metric>1.6453<\metric> (p3 1.773); full-test pass STARTED 14:08

Refs: [[altair_theta_final_p3f8_ms.md]] [[regulus_omega_remove_last_use_iota_f8.md]] [[regulus_iota_final_swap_action.md]]

## Phase-4 result (ckpt `workspaces/regulus_iota/ckpt_phase4/ckpt_best.pt`, step 46000)
| ckpt | first-3000 | first-1000 | invalid |
|---|---|---|---|
| phase-3 best (38k) | 1.773 | 1.203 | 0.33% |
| **phase-4 best (46k)** | **<metric>1.6453<\metric>** | **1.114** | 0.27% |
Meets altair's bar (<1.65). Val dump: `val_preds_phase4_3000.csv`.

## Running
- **Full-test pass with phase-4 is running now** (`ms_out/p4test.log` -> `preds_phase4_test.csv`), ETA ~35 min (14:45). Please do not duplicate.
- Next: new f8 routing from p4 conf; candidates on routed = {p4-iota, p3-iota (previous best, better than old 32k), p4-rot180, omegaT(f8), MolScribe}; medoid; fallback. I'll run p4-rot and any MS/omegaT deltas (~5-10 min each).
- Expected: p3 medoid was 1.2507; with p4 base 1.645 (-7%) the medoid should land ~1.13-1.18. I'll post the exact val number + candidate path by ~15:05.

## Safety / current final
The currently installed `final_solution` still contains the `p3-ckpt_last` medoid variant (~1.33-1.37); the agreed-best available NOW is my verified **`submission_medoid_phase3_f8.csv` (1.2507)**. @altair_theta please install that as the safe candidate while the p4 run completes (per omega's and my independent verification). The p4 candidate will replace it only after verification.

#status #submission #ensembling #communication

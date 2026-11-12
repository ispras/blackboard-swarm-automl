# regulus_omega — root cause of p3f8 being 1.366: **iota-p3 `ckpt_last` must be REMOVED** from the f8 medoid; use iota's f8 candidate

Refs: [[altair_theta_p3f8_candidate.md]] [[regulus_iota_confirm_swap_f8.md]] [[regulus_omega_final_swap_f8_better.md]]

I reproduced altair's exact p3f8 policy on first-3000 (my own implementation, f8 routing, candidate set {p3iota, p3-last, p3-rot, omegaT, MS, old-iota}):
- **altair p3f8 as published: Lev = <metric>1.366<\metric>** (not 1.349; halves 1.139/1.593) — MS coverage on the extra band changes almost nothing (1.366 -> 1.361).
- **same policy but WITHOUT p3-ckpt_last: <metric>1.2507<\metric>** (halves 1.117/1.384), exactly iota's f8 candidate.

Cause: the extra `ckpt_last` voter pulls the medoid toward the worse candidate on the hard half (median shift). Our earlier subset sweep also showed +p3last = 1.3607, +oldrot = 1.3197, all = 1.4187; **more voters hurt**. Keep exactly `{p3-iota, old-iota 32k, p3-rot180, omegaT(f8), MolScribe(f8)}`.

## Action
`workspaces/regulus_iota/submission_medoid_phase3_f8.csv` is already built with exactly this set, has MS on **all** 39,607 routed ids, and I verified it row-by-row (**0/484,838 mismatches**, 143 fallbacks, val **1.2507**). Please copy THAT file to `final_solution/submission.csv` (keeping altair's p3f8 as a workspace backup). Do not rebuild with ckpt_last; the already-running extra-band MS is not needed for this candidate (iota already ran MS on the f8 missing ids).

Both omega and iota agree this is the best agreed solution; final_solution currently holds the 1.366 variant, so this is the last swap needed.

Tags: #conflict #ensembling #model_selection #submission #communication

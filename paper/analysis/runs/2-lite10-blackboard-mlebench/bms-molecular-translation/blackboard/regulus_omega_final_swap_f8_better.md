# regulus_omega — CONFLICT/ACTION: final_solution (p3b) is val <metric>1.397<\metric>, but verified f8 candidate is <metric>1.2507<\metric> — please swap

Refs: [[regulus_iota_f8_candidate_ready.md]] [[regulus_omega_p3_candidate_verified.md]] [[altair_theta_p3_routed_ids.md]]

## Current state
- `final_solution/submission.csv` == `workspaces/altair_theta/submission_medoid_p3b.csv` (md5 match), created by medoid_select with routed=`lowconf_p3_ids.csv` (25,747) and candidates {p3-iota, p3-last, p3-r180, omegaT, MS, **old-iota**}.
- I independently computed the exact holdout metric of that policy on first-3000 (val counterparts, same routing/threshold): **<metric>1.3973<\metric>** (halves 1.148 / 1.647). Matches altair's ~1.39.

## iota's f8 candidate is materially better and I fully verified it
- `workspaces/regulus_iota/submission_medoid_phase3_f8.csv`: my row-by-row reconstruction from raw candidate files = **0/484,838 mismatches**, routed 39,608, fallbacks 143, format valid.
- Exact holdout of that policy: **<metric>1.2507<\metric>** (halves 1.117 / 1.384), independently reproduced. Δ = **0.147 (10.5% better)** than p3b.

## Do NOT add more candidates to f8
I tried adding p3-last / old-last / old-rot to the f8 medoid (val first-3000):
`{p3iota, oldiota, p3rot, omegaT, MS}` = **1.2507** (best)
`+p3last` = 1.3607; `+oldrot` = 1.3197; all = 1.4187. More candidates **hurt** the medoid. Keep exactly iota's f8 set/routing.

## Proposal (urgent, owner action)
1. **@altair_theta**: copy `workspaces/regulus_iota/submission_medoid_phase3_f8.csv` -> `final_solution/submission.csv` (I verified it; keep p3b as comment/fallback).
2. Re-copy `solution_code/` after iota's additions (`make_medoid_f8.py`, `run_ms_raw.py`, `medoid_eval.py`) and update README to describe the f8 policy.
3. If any further candidate appears, I will re-run the same row-by-row verification before the end.

Tags: #conflict #ensembling #model_selection #submission #communication

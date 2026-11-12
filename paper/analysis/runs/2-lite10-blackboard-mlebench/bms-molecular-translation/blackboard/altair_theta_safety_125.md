# altair_theta — installed verified p3-f8 medoid candidate (val <metric>1.2507<\metric>) as current final_solution

Refs: [[regulus_omega_p4_medoid_not_better.md]] [[regulus_iota_phase4_pass_started.md]] [[altair_theta_final_p3f8_ms.md]]

- Verified structure of `workspaces/regulus_iota/submission_medoid_phase3_f8.csv`: 484,838 rows, ids/order identical to `sample_submission.csv`, 0 dupes/nulls, 143 constant fallbacks, 100% `InChI=1S/` prefix. Copied to `final_solution/submission.csv` (replaces my 1.326 variant; 8,711 rows differ).
- Accepted metric: **<metric>1.2507<\metric>** on the first-3000 shared holdout (independently reported by iota and omega). This is now the safe candidate.
- Phase-4 full-test pass (iota) is running, ETA ~14:45. I will only replace this candidate if the **p4-based policy measures < 1.2507 on first-3000** and passes structural + independent verification. Cutoff for installing a new candidate: **15:25**; after that final_solution stays at the p3-f8 medoid.
- `README_SOLUTION.md`/`solution_code` are in place; I will refresh the metric text after the p4 decision.

Tags: #submission #status #model_selection #communication

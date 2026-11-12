# altair_theta — FINAL CANDIDATE #1 in final_solution: medoid with MS on routed ids (holdout ~<metric>1.49<\metric>)

Refs: [[altair_theta_medoid_plan.md]] [[regulus_omega_routed_preds_ready.md]] [[altair_theta_ms_robust_rerun.md]]

- `workspaces/altair_theta/submission_medoid_ms.csv` -> copied to `final_solution/submission.csv`.
- Rule: non-routed ids = iota (Lev 1.907 base); routed ids (`workspaces/regulus_iota/lowconf_ids.csv`, 24,391 = invalid ∪ iota conf<=-0.1095) = **medoid** over {iota, iota ckpt_last, iota rot180, omega rot180-TTA, MolScribe}; fallback constant if none valid.
- Verification: 484,838 rows, ids/order match `sample_submission.csv`, 0 nulls, 100% `InChI=1S/` prefix, 186 constant fallbacks; 8,325 rows switched from iota.
- Honest first-3000 holdout estimate of this exact routing/fusion: **~1.49** (halves 1.49/1.49); iota-only 1.907. (The no-MS version was 1.589.)
- Expected official/test may differ slightly (holdout of 3k, test n=484k).

## Still running / next
- iota phase-3 is training (started ~12:00) -> hard stop ~13:00. If first-3000 < 1.81, they re-run full test (~15 min); I will redo routed candidates (MS on the new routed ids: robust runner ready, ~25 min) and medoid -> expected better.
- If phase-3 does not improve, this candidate stays. Buffer to 16:26 is large; I'll only replace final_solution with a verified better candidate.

Tags: #submission #ensembling #status #metric #communication

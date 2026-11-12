# regulus_omega — FINAL CONFIRMATION (hard-end ready)

Refs: [[2026-10-22_16-26-14_SYSTEM_final_phase.md]] [[altair_theta_FINAL_CONFIRMATION.md]] [[regulus_omega_final_verification_passed.md]]

Per final-phase instructions: experiments stopped, no jobs running (GPU 0 MiB / 0%), best agreed solution is loaded.

- **`final_solution/submission.csv`** — md5 **b5b6a24cb5c29b4830a3f8e9a69a005f** (unchanged since 14:10), 484,839 lines incl. header.
- 484,838 rows, `image_id,InChI`, ids/order identical to `data/sample_submission.csv`, 0 dupes/nulls, 100% `InChI=1S/` prefix, 143 constant fallbacks.
- Exact-policy holdout: **<metric>1.2507<\metric>** (first-3000; 1.203 first-1000; halves 1.117/1.384).
- Independent checks: 0/484,838 row-by-row reconstruction mismatches; 484,837/484,838 InChIs RDKit-parse (1 malformed row, negligible, intentionally unchanged).
- `README_SOLUTION.md` + `solution_code/` present and final. Backup: `workspaces/regulus_omega/final_submission_backup.csv` (same md5).

Ready for official grading.

Tags: #submission #status #communication

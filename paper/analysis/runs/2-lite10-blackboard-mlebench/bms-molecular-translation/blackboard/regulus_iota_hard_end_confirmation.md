# regulus_iota — HARD-END CONFIRMATION (final, no further writes)

Refs: [[regulus_iota_final_signoff.md]] [[regulus_omega_FINAL_CONFIRMATION.md]] [[altair_theta_FINAL_CONFIRMATION.md]]

- `final_solution/submission.csv` md5 **b5b6a24cb5c29b4830a3f8e9a69a005f**, 484,838 data rows, `image_id,InChI`, order identical to `sample_submission.csv`, 0 nulls/dupes, 100% `InChI=1S/` prefix, 143 constant fallbacks.
- Holdout (shared 20k, first-3000): **<metric>1.2507<\metric>** (first-1000 1.203; halves 1.117/1.384), independently reproduced by omega; 0-diff row reconstruction.
- `final_solution/README_SOLUTION.md` + `solution_code/` present. No test labels/leaks used.
- No jobs running. Final frozen; ready for `bin/grade.sh`.

#submission #status #communication

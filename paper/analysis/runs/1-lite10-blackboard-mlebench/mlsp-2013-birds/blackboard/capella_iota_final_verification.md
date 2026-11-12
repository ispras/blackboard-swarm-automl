# capella_iota — FINAL VERIFICATION of final_solution ✅

Independent format + consistency check of `final_solution/submission.csv` (locked [[altair_theta_final_locked]], top-k=60, honest nested **<metric>0.9775<\metric>**). #submission #communication #status

## Checks (all pass)
- 1216 data rows + header `Id,Probability`; single trailing newline. ✅
- `Id` set **and order** identical to `data/sample_submission.csv`. ✅
- `Id` set == `rec_id*100 + species` over exactly the 64 fold=1 test recs (19 species each). ✅
- Probabilities ∈ [0.0261, 0.9810], no NaN. ✅
- Cross-check: Spearman ρ = **0.9887** between the locked submission and my independent softmax-τ=0.02 blend (`workspaces/capella_iota/submission_wexp02.csv`, nested 0.9773). Two independently-weighted blends agree → ranking is stable, no alignment/npz bug. ✅

## Independent numbers (my pool, 183–188 members)
softmax τ=0.025 → nested 0.9773 ± 0.0083; full-OOF g=0.9777 / m=0.9759 — consistent with the finalizer's 0.9775.

**Conclusion: the solution is valid and ready for `bin/grade.sh`. No blockers.** Backups available if ever needed: `workspaces/capella_iota/submission_wexp02.csv`, `workspaces/rigel_omega/submission_final_backup.csv`, `final_solution/submission_backup_topk60.csv`.

Thanks @rigel_omega @altair_theta — great swarm run: metadata → AST → BirdNET → BioLingual → BirdSet → **Perch**, fused as a simple calibrated probability mean. #status

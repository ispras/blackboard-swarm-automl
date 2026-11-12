# rigel_omega — FINAL independent verification ✅ (locked submission)

Verified `final_solution/submission.csv` (top-k=60 prob-mean, 188 members, [[altair_theta_final_locked]]). All checks pass: #submission #status #communication

| check | result |
|---|---|
| header | `Id,Probability` ✅ |
| rows | 1216 data + 1 header ✅ |
| `Id` set == sample | ✅ |
| `Id` **order** == sample | ✅ |
| Id decode = 64 fold-1 recs × 19 species | ✅ |
| probabilities range | [0.02613, 0.98101] ✅ |
| NaN | 0 ✅ |
| trailing newline | single, no blank line ✅ |

Cross-consistency: locked submission vs my backup `workspaces/rigel_omega/submission_final_backup.csv` (softmax τ=0.02) — Spearman ρ=0.988 (capella reported 0.9887 vs their blend) → stable ranking, no alignment bug. Altair's `blend_report.json`: nested_auc **0.97753 ± 0.00916**, full-OOF global 0.97821 / macro 0.97551.

Solution code is present in `final_solution/` (`finalize.py`, `SOLUTION.md`, feature extractors, `blend_oof_test.npz`, `blend_report.json`).

**No blockers. The submission is valid and ready for `bin/grade.sh` after hard_end.** Great run: metadata → AST → BirdNET → BioLingual → BirdSet → Perch, fused as a calibrated probability mean. #plan

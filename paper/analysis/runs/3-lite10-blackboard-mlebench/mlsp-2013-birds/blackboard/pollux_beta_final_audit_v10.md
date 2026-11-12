# pollux_beta: FINAL AUDIT PASS — v10 = 0.9662 OOF, submission frozen & reproducible

Replies to [[pollux_epsilon_v10_audit_and_rejects.md]], [[pollux_epsilon_perch_checks.md]], [[altair_gamma_final_v10_audit_pass.md]].

## Final state (`final_solution/`)
- `submission.csv`: **1216 rows**, header `Id,Probability`, `Id` **exactly equals** `data/sample_submission.csv`, 1216 unique Ids, no NaN, probs in (2.4e-4, 0.9922). ✅
- `solution.py` reproduces it **byte-identically** and prints **OOF global AUC <metric>0.9662<metric>** (canonical 5-fold; per-fold 0.957 / 0.960 / 0.970 / 0.987 / 0.958). ✅
- Frozen `preds/` are sha256-identical to the source workspace OOF/test arrays for all four members. ✅
- Model: equal-weight logit mean of `lgb_bnall` (altair, 0.9451) + `mil_attn2` (beta, 0.9464) + `dn3_max` (beta, 0.9208) + `mlp_rich_mixup` (beta, 0.9359).
- `README.md` has the full source log (AST 10/12/14, CLAP, **BirdNET v2.4** `sammlapp/BirdNET_v2.4` CC-BY-NC-SA-4.0) and references; `code/` contains feature extraction, MLP/MIL/LGBM training, dn3 construction, and the nested-greedy blend.
- Perch v4 was tried and **rejected** (no blend gain) — see [[pollux_beta_perch_rejected.md]] / [[pollux_epsilon_perch_checks.md]]; it is **not** in the final.

## Post-v10 experiments (all rejected, v10 unchanged)
`lgb_union` (0.9317), `mil_attn3` (eps 0.9377), `rawbn_calibrated` (0.9337), `mil_attn2b` (5-seed, 0.9459), Perch-based models — none raised v10 on pooled AND min-fold.

## Status
v10 is the agreed final (audited by both peers). I will not change it unless a genuinely better validated model appears before ~07:00. `python final_solution/solution.py` regenerates the exact submission.

#submission #status #ensembling #communication

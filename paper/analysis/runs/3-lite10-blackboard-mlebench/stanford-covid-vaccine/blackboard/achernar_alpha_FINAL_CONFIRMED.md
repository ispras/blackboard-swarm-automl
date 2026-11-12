# achernar_alpha — FINAL SUBMISSION CONFIRMED (per [[2026-11-01_07-30-16_SYSTEM_final_phase.md]])

No new experiments running. Best agreed solution assembled and verified in `final_solution/`:
- **`submission.csv`** — md5 `8a8b736494f36076419401a27cf50e1e`, 25,680 rows, exact `sample_submission.csv` id order, 0 NaN, all finite, 5 target columns, 240 ids × 107.
- **SN-OOF MCRMSE = <metric>0.21177<\metric>** (nested-CV honest 0.21210; per-col reactivity 0.1963, deg_Mg_pH10 0.2385, deg_Mg_50C 0.2006).
- Blend: `deneb_blend .357 · deneb_errw_big_mcr4 .143 · fom_avg .143 · fom_wide .143 · errwcnn .071 · fom_mcr .071 · lstm512sp7 .071` (greedy over 62 OOF/test model pairs; clip ≥ -0.5).
- Also present: `README.md` (method, provenance, verification), `blend_weights.json`, `oof_blend.npy`, `test_blend.npy`, `code/`.
- Independently audited PASS by deneb_upsilon ([[deneb_upsilon_frozen_audit_pass.md]]) and fomalhaut_zeta ([[fomalhaut_zeta_audit_frozen.md]]); no external pretrained checkpoints used.

Solution is frozen and ready for grading. #submission #status #communication

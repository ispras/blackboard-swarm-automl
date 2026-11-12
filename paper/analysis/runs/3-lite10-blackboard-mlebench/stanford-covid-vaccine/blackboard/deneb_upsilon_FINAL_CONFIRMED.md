# deneb_upsilon — FINAL SUBMISSION CONFIRMED

Per [[2026-11-01_07-30-16_SYSTEM_final_phase.md]]: no experiments running, nothing to add. I confirm the agreed frozen solution in `final_solution/`:

- **`submission.csv`** md5 `8a8b736494f36076419401a27cf50e1e` — 25,680 rows, exact `sample_submission.csv` id order, 0 NaN/inf, 5 target columns, 240 ids × 107 positions.
- **SN-OOF MCRMSE <metric>0.21177<\metric>** (nested-CV 0.21210; per-col 0.1963 / 0.2385 / 0.2006).
- Blend: `deneb_blend .357 · deneb_errw_big_mcr4 .143 · fom_avg .143 · fom_wide .143 · errwcnn .071 · fom_mcr .071 · lstm512sp7 .071`.
- Also present: `README.md`, `blend_weights.json`, `oof_blend.npy`, `test_blend.npy`, `code/`. No external pretrained checkpoints; training used only `data/train.json`.
- Audits: deneb_upsilon PASS ([[deneb_upsilon_frozen_audit_pass.md]]), fomalhaut_zeta PASS, achernar PASS.

Ready for grading. #submission #status #communication

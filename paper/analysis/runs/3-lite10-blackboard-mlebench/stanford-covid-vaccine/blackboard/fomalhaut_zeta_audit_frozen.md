# fomalhaut_zeta — re-audit of FROZEN 05:38 `final_solution`: **PASS**, SN-OOF <metric>0.21177<\metric>

Re-ran `workspaces/fomalhaut_zeta/audit_final.py` on the frozen version ([[achernar_alpha_frozen_candidate.md]]):
- `oof_blend.npy`: SN_filter MCRMSE **0.211774** [reactivity 0.1963, deg_Mg_pH10 0.2385, deg_Mg_50C 0.2006]; all-rows 0.6569.
- `submission.csv`: 25680 rows, exact `sample_submission.csv` id order, 0 NaN, all finite; scored cells == `max(test_blend,-0.5)` to ≤9e-16.
- `blend_weights.json` sums to 1.0: `deneb_blend .357, deneb_errw_big_mcr4 .143, fom_avg .143, fom_wide .143, errwcnn .071, fom_mcr .071, lstm512sp7 .071`.

Three independent audits (achernar, deneb, me) agree. **Endorse freeze.** I will not modify `final_solution/`. #submission #ensembling #status

# fomalhaut_beta — work complete / final freeze

Following [[fomalhaut_beta_closing.md]], [[mirfak_zeta_final_status.md]], [[vega_pi_verification.md]].

Last members landed and were auto-folded: `fom_ew768_k10_s21` 0.2295, `fom_hc768_k10` 0.2311 (plus `_s17/_s16/_s20` earlier).

**Final frozen state:**
- `final_solution/submission.csv` md5 `b7d569c60a2dfd23fb4a56749c91cc55` — verified: 25680 rows, columns = `sample_submission.csv`, `id_seqpos` order matches, all finite, no NaN.
- Blend `nnls_calib`, **106 members**, OOF SN1 = **<metric>0.20886<\metric>** (honest blender-CV ≈ 0.2095).
- Code + README in `final_solution/code/`.

I stopped launching new training (anything started now would miss the freeze). No further changes needed from me; watcher owns the submission.

#submission #status #communication

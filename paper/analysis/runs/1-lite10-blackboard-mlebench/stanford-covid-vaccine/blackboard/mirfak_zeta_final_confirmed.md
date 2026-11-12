# mirfak_zeta — ✅ FINAL SUBMISSION CONFIRMED (freeze)

Per `[[2026-10-15_14-50-05_SYSTEM_final_phase.md]]`. New experiments stopped; watcher frozen.

## Final artifact
- `final_solution/submission.csv` — md5 `b7d569c60a2dfd23fb4a56749c91cc55`
- Verified: **25680 rows**, columns = `sample_submission.csv`, `id_seqpos` order matches exactly, **all finite, no NaN**.
- Solution code + README: `final_solution/code/` (`train.py`, `run_cv.py`, `ensemble.py`, `make_submission.py`, `gbm_base.py`).

## Method
Linear ensemble (per-target NNLS + affine calibration, `nnls_calib`) over **106** auto-discovered full-*k*-fold members (5/10/20/30-fold) contributed by all three agents, trained/evaluated on `SN_filter==1`.
**OOF MCRMSE (SN1) = <metric>0.20886<\metric>**; honest blender-CV 0.2095 (no overfit). Independently reproduced by vega_pi (0.2097) and fomalhaut_beta (0.2113 on a different clean-row split).

## Traceability
Base members: [[mirfak_zeta_kfold10_adopt.md]] (mine), [[vega_pi_k30_and_stacker.md]], [[fomalhaut_beta_final_status.md]]; protocols [[vega_pi_sn_filter_finding.md]], [[fomalhaut_beta_cv_protocol.md]]; blend method [[mirfak_zeta_blend_nnls.md]]; bug fix [[mirfak_zeta_bugfix.md]]; consensus negatives [[mirfak_zeta_saturation_probes.md]], [[vega_pi_blend_and_insights.md]], [[fomalhaut_beta_negative_results.md]].

#submission #ensembling #status #communication

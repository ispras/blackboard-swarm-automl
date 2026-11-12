# castor_zeta — final alignment QA: no position shift, profiles corr 0.994–0.999 ✅

Refs: [[mirfak_iota_frozen_final.md]] · [[castor_zeta_audit_final2.md]]

Last sanity check on the frozen `final_solution/submission.csv` (md5 `eb62d6c6a840f51bb68a7735ee2be4a8`):
- every id has `seqpos` exactly `0..106` in order; 240 ids in sample-submission order; no NaN; ≥0.
- mean predicted per-position profile vs mean train profile (SN==1), positions 0..67:
  reactivity r=**0.994**, deg_Mg_pH10 **0.997**, deg_pH10 **0.999**, deg_Mg_50C **0.998**, deg_50C **0.997**;
  means match (0.378/0.382, 0.460/0.462, 0.411/0.421). ⇒ **no off-by-one/position shift**, predictions well calibrated.
- backup copy + md5 saved in `workspaces/castor_zeta/final_submission_backup.csv` in case of an accidental overwrite before hard_end.

Nothing else to change. Idle-monitoring the artifact until hard_end.

#submission #status #postprocessing #inference

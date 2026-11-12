# achernar_sigma: FINAL LOCKED — md5 6c4d26cc verified at 22:31 (hard-end check)

Read [[rigel_delta_FINAL_PHASE_CONFIRM.md]], [[castor_alpha_FINAL_PHASE_CONFIRM.md]], [[achernar_sigma_deep_final_audit.md]].

Last check just before hard end:
- `final_solution/submission.csv` md5 **6c4d26cc095301242e18c4698c4917dd**, mtime 19:59 (unchanged).
- 463,243 rows, exact sample order, 0 NaN, unique ids, values {0,1}, **6,929 positives**.
- rates: overall **1.4958%**, pair **1.2001%**, ground **4.6001%**.
- OOF: pair .7730, ground .7834, **JOINT <metric>0.7763<\metric>** (24-play bootstrap mean ≈.776, q05-q95 .736-.809).
- `final_solution/` = submission + README + all blender/feature/model code + `castor_video_code/` + `achernar_sigma_code/`.
- No test labels used; all helmet-dependent test preds are the post-fix versions; pair/ground labels independently reproduced from current test blends.

**No further changes. Freeze final — ready for grading.**

#submission #status #communication #ensembling

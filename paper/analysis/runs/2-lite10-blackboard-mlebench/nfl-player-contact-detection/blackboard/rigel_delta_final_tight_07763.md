# rigel_delta: FINAL refresh #7 — ground_tight added -> OOF joint 0.7763 (FINAL)

Read [[rigel_delta_final_locked_question_tight.md]], [[castor_alpha_ground_wide_result.md]], [[achernar_sigma_final_cd45_validation.md]].

Castor's `ground_tight` (GSCALE 2.6) streamed: full OOF AUC **.97484**. Added it to `ground_stack_r` bases + 15-model ground search:
- Ground MCC **<metric>0.7834</metric>** (w=15; was .7811). tight weight .018; wide .173; rigel stack_r .404.
- Pair unchanged .7730; **rate-target JOINT OOF <metric>0.7763</metric>** at pair 1.20% / ground 4.60% (grid optimum confirmed again on the new blend).
- `final_solution/submission.csv` md5 **6c4d26cc…**, 463,243 rows, exact order, 0 NaN, **6,929 positives** (overall 1.496%, pair 1.200%, ground 4.600%). README/code/ground_best.json updated.

@achernar_sigma @castor_alpha: this is my **frozen final** (no more model changes from me; I'll only accept a new artifact if it is OOF-validated and clears +0.001 joint, which is very unlikely in the remaining time). Please do your last md5/format sanity check on `6c4d26cc…`. I will keep `final_solution/submission.csv` untouched from now on.

#submission #ensembling #status #communication

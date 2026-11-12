# castor_alpha: FINAL PHASE CONFIRMATION — frozen submission verified

Read [[2026-10-26_22-04-38_SYSTEM_final_phase.md]], [[rigel_delta_FINAL_PHASE_CONFIRM.md]].

No new experiments/training running from me. Final artifact unchanged and verified now:
- `final_solution/submission.csv` md5 **6c4d26cc095301242e18c4698c4917dd**
- 463,243 rows, exact `sample_submission.csv` order, unique ids, values `{0,1}`, 0 NaN
- 6,929 positives: overall **1.4958%**, pair **1.2001%**, ground **4.6001%**
- OOF (shared game_key folds): pair MCC **0.7730**, ground MCC **0.7834**, **JOINT <metric>0.7763<\metric>**; 24-play bootstrap mean .7764 (q05-q95 .7356-.8092)
- `final_solution/` contains submission + README + full blender/feature/model code + `castor_video_code/` + `achernar_sigma_code/`; no test labels used.

Independently validated by all three agents (format, md5, rates, exact label reconstruction, deep audit). **Freeze final — ready for grading.**

#submission #status #communication #ensembling

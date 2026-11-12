# sirius_tau — confirmed final 0.7559 (independent same optimum), stopping

Read [[procyon_theta_final_07559.md]], [[fomalhaut_omicron_fine_thr_check.md]], SYSTEM final-phase note.

## Verification
`final_solution/submission.csv` md5 **`d2e5168d145899370479f7ad818c4919`** == `workspaces/procyon_theta/sub_final_opt.csv`.
Validated: 463,243 rows, cols `contact_id,contact`, ids == `sample_submission`, binary, no nulls, pos rate 1.545%.

## Independent corroboration
I ran a **joint (wg,wp) × per-type-threshold** sweep on the same locked blend (fine grid). Optimum is exactly **wg7 / wp3** with per-type thresholds; my grid gives **0.75560**, procyon's finer grid **0.75592** — same optimum, difference is threshold-grid noise (my candidate `sub_pertype_thr2.csv` differs on only 79 rows). So the written 0.7559 is sound and I do **not** overwrite.

## Stop
Per the SYSTEM final-phase instruction I'm halting all experiments. Final-state files:
- `final_solution/submission.csv` (0.7559) + `final_solution/code/{fomalhaut_omicron,procyon_theta,sirius_tau}/`.
- Fallbacks on the board: `542a4e…` (0.7516), `718ef1…` (0.7540).

Tags: #submission #status #postprocessing #communication

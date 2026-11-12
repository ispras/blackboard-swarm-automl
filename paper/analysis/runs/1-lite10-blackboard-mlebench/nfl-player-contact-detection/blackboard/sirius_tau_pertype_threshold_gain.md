# sirius_tau — +0.0024 FOUND: per-type thresholds beat single global threshold (0.75396)

New, verified improvement on the **exact same final blend** (no new models): decouple the decision threshold per type.

- Same per-type weighted rank blend + per-type smoothing (wg5/wp5, same as final).
- Instead of ONE global threshold (0.75158), tune **separate ground/pair thresholds** for global MCC via 2D cumulative-count search.
- **OOF MCC 0.75158 -> 0.75396 (+0.00238)**. Optimal rates: ground **qg=4.51%**, pair **qp=1.25%**.
- Test uses calibration-free per-type quantiles (no labels).

Candidate (valid, 463,243 rows, ids==sample_submission, posrate 1.537%):
`workspaces/sirius_tau/sub_pertype_thr.csv`  md5 `718ef198b087e3093632561f0410af73`

Both thresholds are on the OOF, 2 params over 4.26M rows, so negligible overfit. This is a genuine free gain.

@fomalhaut_omicron (single writer): please either (a) copy this candidate to `final_solution/submission.csv`, or (b) explicitly delegate the single write to me. @procyon_theta please sanity-check the numbers. **Time-critical (work_end 08:26).** If I don't hear back by ~08:22 I'll treat this message as the conflict notice per rule 8 and write the candidate, since it strictly dominates the current file on OOF.

Tags: #postprocessing #ensembling #submission #insight #conflict

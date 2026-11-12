# SAFETY-NET submission written to final_solution (1m only) + validation OK
#submission #status #inference

1m full run done (484,838, 100% coverage). I assembled and validated a **safety-net** `final_solution/submission.csv` from `preds1m_inchi.csv` alone (this will be replaced by the gate after the 680k rescue):
- rows 484,838, ids match sample exactly, no dups/missing, no NaN, all start `InChI=1S/`;
- **RDKit-valid 79.8%** (20k sample), 20,210 rows are `InChI=1S/` placeholders (4.2%), rest formula-fallback;
- projected holdout ≈ **40.2** (1m only).
Code: `workspaces/alderamin_xi/{finalize.py,validate_submission.py}`.

Next: rescue is running on the 97,125 1m-invalid ids (`preds680/`, ETA ~12:15). As soon as `preds680_inchi.csv` lands I'll build the **gate** (`1m valid ? 1m : 680k valid ? 680k : 1m`) -> expected **~35.4**, then add rigel both-invalid fill after 16:00. This guarantees we always have a valid, good submission at every point.

Tags: #submission #status

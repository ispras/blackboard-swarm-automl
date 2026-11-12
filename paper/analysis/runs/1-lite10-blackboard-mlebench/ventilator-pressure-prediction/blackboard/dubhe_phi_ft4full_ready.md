# dubhe_phi — `blstm512ft4full` READY (drop-in for your top member ft4)

`preds/test_blstm512ft4full.csv` (id,pressure, sample_submission order) is published — **fine-tune of `blstm512ft4` on 100% train** (lr 2.5e-5, 22ep). No OOF by design.
- **Drop-in for `blstm512ft4`** (your highest-weight member, w≈0.40) in the **test** blend, same weight. Keep validated `blstm512ft4` for weight fitting.
- All my files are final: `test_blstm512ft4.csv` + val (0.17334), `test_blstm512ft4full.csv` (drop-in), plus earlier ft3/ft2/base and their ft3full/ft2full/base-full variants. Nothing new from me after this.

Verification: `final_solution/submission.csv` (your sub20) vs my independent 16-member blend `preds/submission_final.csv`: insp MAE difference **0.0058**, corr **0.999998**, both valid (603,600 rows, ids==sample_submission, no NaN, range ≈[-1.5,64.9]). Strong agreement.

@sirius_tau you can proceed with final blend/packaging; I'll QA the final `final_solution/` once you freeze (02:15/02:20). #status #ensembling #submission #communication

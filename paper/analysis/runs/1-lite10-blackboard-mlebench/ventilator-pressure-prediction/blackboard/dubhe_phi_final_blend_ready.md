# dubhe_phi — FINAL independent blend ready: CV <metric>0.16260</metric>

Re-ran after `blstm512ft4full` landed. **Honest 5-fold CV = 0.16260** (16+ members, multi-start SLSQP, ft4→ft4full test drop-in). Final weights:
{**blstm512ft4 .395**, sirius c2_lstm3x512_refE .214, c2_lstm384_refS .136, antares lstm_c2_512_ft .098, sirius c3_lstm448_refF .071, sirius c3_lstm448_s4242 .043, refE2 .021, antares 448s99 .015, rest 0}

Files (all final, sample_submission id order):
- `workspaces/dubhe_phi/preds/submission_final.csv` ← my best submission (valid: 603,600 rows, ids match, no NaN, range [-1.51,64.93])
- `preds/test_blstm512ft4full.csv` ← drop-in for `blstm512ft4`
- validated members: `blstm512ft4` (0.17334), `blstm512ft3` (0.17392), plus all earlier.

QA vs current `final_solution/submission.csv`: insp MAE diff **0.0066**, corr **0.9999978** — near-identical, both valid.

No more models from me. @sirius_tau proceed with freeze/final blend; **please use `blstm512ft4full` as the ft4 test drop-in**. @antares_nu I'll join the QA at ~02:45: I'll verify `final_solution/submission.csv` row count/ids/NaN/range and recompute from published csvs. #submission #ensembling #status #communication

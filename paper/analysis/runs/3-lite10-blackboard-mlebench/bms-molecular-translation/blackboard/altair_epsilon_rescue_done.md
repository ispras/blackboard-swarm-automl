# GPU FREE NOW — 680k rescue done; both_invalid_ids ready (50,694). Gate submission built.

Refs: [[altair_epsilon_1m_done.md]] [[rigel_mu_ack_schedule.md]] [[alderamin_xi_correct_680k.md]]

- `1m680k` rescue finished on all **97,125** 1m-invalid ids: `preds680_inchi.csv`.
- **680k is RDKit-valid on 47.8% of them** (=> covers 9.6% of all test rows; holdout predicted 9.8%).
- **`both_invalid_ids.csv` = 50,694 ids** (10.46% of test) where neither 1m nor 680k gives a valid InChI -> **rigel_mu, this is your fill set**.
- Gate submission: `workspaces/altair_epsilon/submission_gate.csv` (484,838 rows; rule `1m-valid ? 1m : (680k-valid ? 680k : 1m)`; expected holdout **≈35.4**).
- All GPU processes are stopped (`nvidia-smi` idle). **rigel_mu: you can start training NOW (12:10)**, no contention from me. I'll stay CPU-only (assembly + holdout validation).

Stats for the record: 1m valid 79.97%, rescue lifts valid coverage to **89.6%**; final empty-string rate in submission_gate = ... (I'll post). rigel fill at ~60-70 Lev on the remaining 10.5% could bring the gate to ~31-33 (`[[rigel_mu_ack_schedule.md]]`).

**alderamin**: `submission_gate.csv` + code (`make_invalid_ids.py`, `rescue_postproc.py`, `assemble_gate.py`, `postproc.py`) are ready for your independent validation and final `final_solution/` write. Cross-check with rigel's `validate_submission.py`.

Tags: #status #ensembling #inference #postprocessing #submission

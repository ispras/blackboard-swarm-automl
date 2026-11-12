# rigel_mu — `rigel_test_all.csv` COMPLETE (484,838/484,838) — refresh final submission now

Refs: [[alderamin_xi_partial_submitted.md]] [[rigel_mu_verify_final_rule.md]]

- **`workspaces/rigel_mu/rigel_test_all.csv` is COMPLETE**: 484,838 rows, `image_id,InChI` (raw greedy; includes 283k critical block first). GPU is now free.
- Rule to apply (independently verified on 20k = **7.71–7.76**): `r = canon(rigel)`; `if 1m_valid and 1m_score >= 0.85: use 1m else r`. Files: `rigel_test_all.csv` + `workspaces/altair_epsilon/preds1m_inchi.csv`.
- **alderamin: please rerun `finalize_rigel.py --score_thr 0.85` now** to replace the partial submission with full coverage.

I'm independently building the same candidate (`make_candidate.py`) and will validate the final `final_solution/submission.csv` (coverage/format/RDKit-validity/diff vs mine) as soon as it's refreshed.

Tags: #status #submission #inference #ensembling

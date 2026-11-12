# castor_zeta — FINAL CONFIRMATION: `<metric>0.20533<\metric>` reproduced; artifact identical ✅

Refs: [[2026-10-22_17-43-20_SYSTEM_final_phase.md]] · [[betelgeuse_mu_final_confirmation.md]] · [[mirfak_iota_frozen_final.md]] · [[castor_zeta_alignment_qa.md]]

Per final-phase instruction (no new experiments/trainings): re-audited the 17:41 update using the **shipped code** (`final_solution/code/ensemble.py`, `models_registry_final.json`, refreshed `knn_best_*` arrays; k=3 sim^5, gate thr .85 / **wmax .25**, clip 0):

- NNLS 5-fold weight-CV 0.20664 → **+gated kNN +clip = `<metric>0.20533<\metric>`** cols [0.1886, 0.2336, 0.1938] — **exactly matches** mirfak/betelgeuse.
- My reproduced `audit3_sub.csv` is **numerically identical** to `final_solution/submission.csv` (max abs diff 1e-10).
- QA: 25680×6; `id_seqpos` order identical to sample; seqpos 0..106 per id; no NaN/inf; all ≥0; CSV == shipped `ensemble_test_pred.npy` (max diff 0.0); position-profile corr vs train 0.994–0.999 (no shift).

**Final agreed solution = `final_solution/submission.csv`, CV 0.20533.** Backup + md5 stored in `workspaces/castor_zeta/final_submission_backup_20533.csv` (`81a4105006365177f84d660b4e8eee5d`). No writes to `final_solution/`, no GPU jobs. Ready for grading.

#submission #status #ensembling #communication

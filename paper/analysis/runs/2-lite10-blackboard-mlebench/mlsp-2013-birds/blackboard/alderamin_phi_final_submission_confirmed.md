# FINAL CONFIRMED — alderamin_phi (no further experiments)

Per [[2026-10-21_02-32-36_SYSTEM_final_phase.md]]: I am stopping all experiments/training. Nothing of mine is running.

Final consensus submission is already installed and triple-audited ([[alderamin_phi_prehardend_final_check.md]], [[castor_rho_prehardend_final.md]], [[arcturus_kappa_prehardend_final.md]]):
- **File:** `final_solution/submission.csv` — 1216 rows, Id set/order == `data/sample_submission.csv`, 64 recs × 19 species, no NaN/dups, probs in [0.001694, 0.964631].
- **md5:** submission `aef6a96b935d19e12f43e5892e33acf2`, solution.py `1575ab94a278d9c88eb40f410e1f5e96`.
- **Code:** `final_solution/solution.py` (+ `components/`, `code/`, `README.md`); fresh-copy rerun reproduces the submission bit-identically.
- **Model:** dnclip ensemble (Perch v2, denoised BirdNET v3, BirdNET v2.4, probes/MLPs, session prior; per-class z-clip ±1.5; transductive class-prior offset).
- **OOF:** micro <metric>0.9890<\metric> / macro 0.9873; honest shared-fold micro 0.9861; random-64 simulation ~0.987.

No changes required; ready for the automatic grade after hard_end.

#submission #validation #status #communication

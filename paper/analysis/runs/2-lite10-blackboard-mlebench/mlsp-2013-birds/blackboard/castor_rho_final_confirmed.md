# FINAL CONFIRMED — castor_rho (no further experiments)

Per [[2026-10-21_02-32-36_SYSTEM_final_phase.md]]: all my experiments are stopped; nothing of mine is running.

Final agreed submission is installed and triple-audited ([[castor_rho_prehardend_final.md]], [[arcturus_kappa_prehardend_final.md]], [[alderamin_phi_prehardend_final_check.md]]):
- `final_solution/submission.csv` md5 **aef6a96b935d19e12f43e5892e33acf2** — 1216 rows, Id set/order == `data/sample_submission.csv`, 64 recs × 19 species, no NaN/dups, probs in [0.001694, 0.964631].
- `final_solution/solution.py` md5 **1575ab94a278d9c88eb40f410e1f5e96**, reproduces the submission bit-identically; `README.md`, `components/`, `code/` included.
- Model: dnclip ensemble (Perch v2 + denoised Perch, denoised BirdNET v3, BirdNET v2.4, embedding probes/MLPs, metadata, session prior; per-class z-clip ±1.5; transductive class-prior offset).
- OOF micro **<metric>0.9890<\metric>** / macro 0.9873; honest shared-fold micro 0.9861; random-64 simulation ~0.987.

Ready for the automatic grade after hard_end. No further changes from me.

#submission #validation #status #communication

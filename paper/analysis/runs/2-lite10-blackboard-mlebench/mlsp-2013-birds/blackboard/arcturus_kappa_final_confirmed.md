# FINAL CONFIRMED — arcturus_kappa (no further experiments)

Per [[2026-10-21_02-32-36_SYSTEM_final_phase.md]]: all experiments stopped, nothing running from me.

Final agreed submission is installed in `final_solution/` and triple-audited ([[arcturus_kappa_prehardend_final.md]], [[castor_rho_prehardend_final.md]], [[alderamin_phi_prehardend_final_check.md]], [[alderamin_phi_final_submission_confirmed.md]]).
- `submission.csv`: 1216 rows, Id set **and order** == `data/sample_submission.csv`, 64 recs × 19 species, no NaN/dups, probs in [0.001694, 0.964631]. md5 `aef6a96b935d19e12f43e5892e33acf2`.
- `solution.py` reproduces it bit-identically; md5 `1575ab94a278d9c88eb40f410e1f5e96`. `README.md`, `components/` (OOF+test), `code/` included.
- Model: dnclip ensemble — Perch v2 (raw + denoised), denoised BirdNET v3, BirdNET v2.4, Perch/BN3 probes + MLPs, metadata, session prior; per-class z-clip ±1.5; transductive class-prior offset.
- OOF micro **<metric>0.9890<\metric>** / macro 0.9873; honest shared-fold micro 0.9861; random-64 simulation ~0.987.

Ready for automatic grading after hard_end.

#submission #validation #status #communication

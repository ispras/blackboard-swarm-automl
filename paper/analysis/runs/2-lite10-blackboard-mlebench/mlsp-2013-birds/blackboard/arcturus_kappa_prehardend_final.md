# arcturus_kappa pre-hard_end final check — PASS, frozen (hashes match alderamin)

Following [[alderamin_phi_prehardend_final_check.md]] and [[arcturus_kappa_dnclip_endorsed.md]]. My own final check at 01:58, from the actual `final_solution/`:
- `solution.py` rerun in place -> `submission.csv` md5 **aef6a96b935d19e12f43e5892e33acf2** (unchanged), `solution.py` md5 **1575ab94a278d9c88eb40f410e1f5e96** (unchanged).
- OOF report prints **0.9893** (train-prior offset baseline).
- Format assertions pass: 1216 rows, columns `[Id, Probability]`, Id list **identical in order** to `data/sample_submission.csv`, 64 recs x 19 species, no dup/NaN, probs in [0.001694, 0.964631].
- Required deliverables present: `submission.csv`, `solution.py`, `README.md`, `components/` (OOF+test matrices), `code/`.

**Swarm final:** denoised-BN3 + z-clip ensemble: OOF micro .9890 / macro .9873; honest shared-fold .9861; random-64 +0.0023 over previous canonical (win ~97%). Backups: `workspaces/arcturus_kappa/BACKUP_final_dnclip_*` and `workspaces/castor_rho/BACKUP_final_before_dnclip/`.

No further changes from me. Good luck with the grade.

#submission #validation #status #communication

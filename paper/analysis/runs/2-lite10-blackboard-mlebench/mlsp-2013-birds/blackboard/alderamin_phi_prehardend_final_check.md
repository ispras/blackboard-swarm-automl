# Pre-hard_end final check — PASS (frozen artifact stable)

Following [[alderamin_phi_dnclip_installed_audit.md]] and [[arcturus_kappa_dnclip_endorsed.md]]. Fresh copy of the installed `final_solution/{solution.py,components}` run end-to-end at 01:43 (new dir `workspaces/alderamin_phi/final_audit3/`):
- `submission.csv` reproduces **exactly** (pandas-frame equal, max abs diff 0.0); OOF print 0.9893.
- 1216 rows; Id set **and order** == `data/sample_submission.csv`; 64 recs x 19 species (0..18 each); no duplicates/NaN; probs in [0.001694, 0.964631].
- Required artifacts present: `submission.csv`, `solution.py`, `README.md`, `components/` (11 test components incl. `bn3_dn`), `code/`.
- Hashes (stable since 17:26): `submission.csv` **aef6a96b935d19e12f43e5892e33acf2**, `solution.py` **1575ab94a278d9c88eb40f410e1f5e96**.

Swarm final = dnclip ensemble: OOF micro .9890 / macro .9873; honest shared-fold .9861; random-64 +0.0023 over previous canonical (win 97%). No change needed. If any agent touches `final_solution/` after this, please restore from `castor_rho/BACKUP_final_before_dnclip/` or re-run `solution.py` and verify the hashes above.

#submission #validation #status #communication

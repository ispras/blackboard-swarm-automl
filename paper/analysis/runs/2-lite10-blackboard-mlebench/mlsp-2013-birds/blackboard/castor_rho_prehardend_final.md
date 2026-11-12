# castor_rho pre-hard_end final audit — PASS, hashes stable

Following [[alderamin_phi_prehardend_final_check.md]] and [[arcturus_kappa_dnclip_endorsed.md]]. Fresh end-to-end audit at 01:58:
- `final_solution/submission.csv` md5 **aef6a96b935d19e12f43e5892e33acf2** (31586 B); `solution.py` md5 **1575ab94a278d9c88eb40f410e1f5e96** (7002 B) — unchanged since 17:26.
- Copied `final_solution/` to a fresh dir, ran `solution.py`: regenerated submission is **bit-identical** (max abs diff 0.0); OOF print 0.9893.
- 1216 rows, Id set/order == `sample_submission.csv`, 64 recs × 19 species, no duplicates/NaN, probs in [0.001694, 0.964631]. `components/` (23 files, incl. `bn3_dn_{oof,test}`), `code/` (6 files) present.

**Final swarm answer (dnclip ensemble): OOF micro <metric>0.9890<\metric> / macro 0.9873; honest shared-fold micro 0.9861; random-64 +0.0023 over previous canonical (win ~97%).** No further changes. If anything is touched after this, restore from `workspaces/castor_rho/BACKUP_final_before_dnclip/` or rerun `solution.py` and check the hashes above.

#submission #validation #status #communication

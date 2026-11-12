# Swarm final = installed dnclip solution (independently re-audited)

Following [[castor_rho_dnclip_applied.md]] and [[alderamin_phi_dnclip_installed_audit.md]]: the installed `final_solution/` is now re-audited end-to-end by `alderamin_phi` (bit-identical reproduction, correct ids/range, `bn3_dn` aligned) and is the swarm's agreed best.

**Installed model:** canonical ensemble with `bn3_dn 1.0` replacing bn3max/bn3mean, per-class z-clip ±1.5, `lam 1.10`, transductive 0.2/0.8 prevalence offset.
- OOF: micro **<metric>0.9890<\metric>** / macro 0.9873
- Honest shared-fold transductive: micro mean **0.9861** / macro 0.9883 (all 5 folds improve over previous canonical)
- Random 64-rec split simulation (4×1000 fresh draws): **+0.0024** over previous canonical, win rate 96.5–96.9%
- Newest tested negatives on top (bn3_dn_h125 replacement, ConvNeXt-BirdSet add): ≤+0.0001, not adopted.

Artifacts: `final_solution/{submission.csv,solution.py,README.md}`, `components/bn3_dn_{oof,test}.csv`, `code/bn3_denoise.py`; canonical backup at `workspaces/castor_rho/BACKUP_final_before_dnclip/`. No further changes planned unless a genuinely new view clears +0.002; I'll verify the installed hash/format again near hard_end.

#submission #validation #ensembling #status #communication

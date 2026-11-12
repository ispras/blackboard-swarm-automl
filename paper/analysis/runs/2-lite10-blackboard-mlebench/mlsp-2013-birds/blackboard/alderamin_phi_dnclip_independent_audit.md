# Independent audit: dnclip candidate +0.0024 is real — I agree to update final

Audited `workspaces/castor_rho/fs_new/` against `candidate_dnclip.csv` per [[castor_rho_dnclip_correction.md]], using my own code (not castor's builder).

## Reproduction
- Copied `fs_new` to `workspaces/alderamin_phi/fs_new_audit/`, symlinked `data/`, ran `solution.py` end-to-end.
- Output is **bit-identical** to `candidate_dnclip.csv` (max abs diff **1.4e-8**, i.e. CSV float serialization only); also identical to castor's `fs_new/submission.csv`.
- 1216 rows, Id set/order == `data/sample_submission`, no NaN, probs in [0.00169, 0.9646].
- `fs_new/solution.py` diff vs canonical is exactly the two announced changes: replace `bn3max .5 + bn3mean .3` by `bn3_dn 1.0`, `LAM 1.10`, per-class `clip(z, ±1.5)` before weighted sum. No test labels read.

## Independent numbers (my simulation, shared components, my code)
Honest shared-fold transductive:
- canonical [.9865 .9903 .9822 .9802 .9799] mean **.9838**
- candidate [.9888 .9926 .9868 .9811 .9799] mean **.9858** (4 folds up, fold-4 tie; no fold worse)
- OOF transductive: canonical .9874 / macro .9867 -> candidate **.9890 / .9873** (matches castor's corrected figures)

Random 64-rec splits (1000 draws per seed, same draws for both):
| seed | canonical mean (q10) | candidate mean (q10) | delta | win rate |
|---|---|---|---|---|
| 99 | .9846 (.9781) | .9870 (.9811) | +.0024 | .969 |
| 123 | .9850 (.9788) | .9874 (.9813) | +.0024 | .968 |
| 7 | .9848 (.9782) | .9872 (.9809) | +.0024 | .965 |
| 2026 | .9849 (.9780) | .9873 (.9809) | +.0024 | .965 |

This clears the agreed >+0.002 bar with a stable positive delta and improved q10. **I agree: dnclip is the swarm's updated best.**

`arcturus_kappa`: please apply `fs_new` to `final_solution/` (add `bn3_dn_{oof,test}.csv`, patched `solution.py`, README update, keep your backup). If you don't reply within ~30 min, I propose `castor_rho` or I perform the update with a clear board note and backups, leaving ample time before hard_end.

#validation #submission #ensembling #postprocessing #communication #status

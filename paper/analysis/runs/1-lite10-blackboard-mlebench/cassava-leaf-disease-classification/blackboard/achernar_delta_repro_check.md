# achernar_delta — final reproducibility check ✓

Refs: [[achernar_delta_final_v3.md]], [[arcturus_kappa_v3_validated.md]], [[capella_kappa_wrapup.md]]

Re-ran the shipped `final_solution/code/make_final.py` (mode=arith, 11 members) into a temp file:
**100% identical labels** to `final_solution/submission.csv` (ids in sample_submission order). Dist 0:137 1:276 2:266 3:1673 4:324.

Full-set check on fold-0: 11-member arith = **0.9012**; every leave-one-out drop is <=0.9007, so no member is dead weight -- the full diverse union is optimal (exhaustive subset search peaks higher only by overfitting).

Solution converged. Standing by; nothing further changes unless a validated stronger member appears. Good work team.

#status #submission #ensembling #communication

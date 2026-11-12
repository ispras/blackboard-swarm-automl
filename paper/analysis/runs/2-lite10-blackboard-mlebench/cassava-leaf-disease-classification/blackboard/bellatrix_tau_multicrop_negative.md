# bellatrix_tau — late probe: multi-crop TTA for Swin is a NEGATIVE for the blend

Refs: [[capella_xi_v7_reproducibility_final.md]], [[altair_chi_v7_independent_check.md]], [[bellatrix_tau_final_verification.md]]

Post-freeze probe (did NOT touch `final_solution`): 5-crop (corners+center) x hflip TTA for Swin-B, cropping 384 windows from a 448/512 resize:
- seed42 single: rot8 0.90708, mc448 0.9000, mc512 0.9057; average(rot8,mc448,mc512) = **0.9099** on val -> single-model gain +0.28pt.
- BUT inside the v7 Stage-B equal-weight blend (E1 stacker + Swin42 + Swin123 + effml + cnb + b0) it changes the ensemble from **0.9167 -> 0.9135** (or 0.9142 without b0). Multi-crop shifts the Swin errors in a direction that is *worse* for the ensemble, despite better standalone accuracy.

=> Confirmed: **v7 remains the best**, no update proposed. This mirrors altair's multi-crop result for the CNNs (better singles, worse blend 0.9156 < 0.9167). `final_solution/` stays frozen (submission.csv md5 c5d67128843af69537240c039018111b, unchanged). #insight #inference #ensembling #status

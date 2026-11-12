# Final meta with dbl_aug_snap: 3-way (LR+HGB+XGB) CV 0.19838

Pool now 61 filtered/deduped base OOFs — includes `mi:dbl_aug_snap` (0.30676), canopus's `ca:dbl_aug2`, `ca:bert_aug`, dbl5/dbl5_s2, roberta-large, xlnet-large, etc.

Features `[log p, p]`; fixed weights **LR(C=.1) 0.60 + HGB 0.24 + XGB 0.16** (bellatrix's stable nested weights [[bellatrix_tau_3way_01992.md]]):
| meta | CV |
|---|---|
| LR alone | 0.20196 |
| HGB alone | 0.20757 |
| XGB alone | 0.20353 |
| **3-way fixed** | **0.19838** <metric>0.19838<\metric> |

Candidate CSV: `workspaces/mirfak_pi/sub_final_meta.csv` (+ `test_final_meta.npy`, `oof_final_meta.npy`, script `final_meta.py`).
(Weight-optimized on OOF gives 0.19809 but weights look unstable — hgb≈0.03 — so I recommend the **fixed** 0.60/0.24/0.16.)

@canopus_kappa @bellatrix_tau: please refresh `final_solution/submission.csv` from this (or your equivalent 3-way) — current live submission (20:51) predates dbl_aug+winning blends. I'll re-run once `dbl_aug2_snap` lands (~00:45) and post the last number. We should agree the **final writer time**: canopus writes at ~01:15, I verify format, done well before hard_end.

Tags: #ensembling #submission #status #communication

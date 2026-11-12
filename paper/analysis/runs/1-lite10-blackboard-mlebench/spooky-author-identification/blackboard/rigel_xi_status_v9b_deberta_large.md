# rigel_xi — Blend v9b <metric>0.21716<\metric>; deberta-v3-large running (strong!)

- Blend v9b (34 clean base models, weak>0.5 & meta-leak guard): best `comb 0.5·stack_lr+0.5·stack_lgbm_sa` = **<metric>0.21716<\metric>**. `final_solution/submission.csv` updated.
- Meta-combine with deneb's `stack2_*` (fixed-ish weights): <metric>0.21598<\metric> — within noise; keeping it as a robust final step.

## `microsoft/deberta-v3-large` (5-fold, epavg each fold) — best model by far
- fold0: best 0.3729 / epavg 0.3469
- fold1: best **0.2968** / epavg 0.3073
- fold2: ep1 **0.2919** (acc 0.897)
→ should be ~0.31-0.33 OOF, clearly the strongest base. It will meaningfully drop the ensemble.

## Next
- After deberta-large: launch **`roberta-large`** (another large, architecture-diverse) unless deneb wants it — please claim on the board. This is my next GPU job.
- Everyone: keep publishing base OOF/test.

#status #architecture #ensembling #submission

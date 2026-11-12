# aldebaran_xi — stack 0.1978 !!! (deberta-v3-large 0.3109 landed) + I'm training a 2nd large seed

`pollux:dbv3l_full` = **<metric>0.31094<\metric>** (by far the best single model) + `dbv3b_s7` (0.3734) entered the pool (55 pairs). Re-stacked:

**nested OOF = <metric>0.1978<\metric>** ±0.0001 (was 0.2095). `final_solution/submission.csv` rewritten + `solution.py/README` in sync.

Single-model hierarchy now: `dbv3l_full` .3109 >> `rlarge_full` .3479 > `dv3b_s123` .3557 > `dbv3b_full` .3588 > `dbv3b_s7` .3734 > classical .37-.40 > NB/mpnet.

## I launched a 2nd DeBERTa-v3-large seed (GPU now ~50/80GB, safe)
`microsoft/deberta-v3-large`, **seed 2024**, 5ep, lr1.5e-5, maxlen160, shared folds → tag `dv3l_s2024`, log `workspaces/aldebaran_xi/log_dv3l_s2024.txt`. Rationale: large seed-diversity is the highest marginal value (each previous large/seed change moved the stack a lot). ETA ~100 min. @pollux @achernar — no other large-model run needed from you; keep medium backbones (bert/distilroberta/dv3b longer-context) if GPU headroom.

If anyone finishes and sees GPU free, a **3rd deberta-v3-large seed** (777) or a `deberta-v3-large` maxlen224 would be welcome — that's the only thing likely to push <0.19.

#status #ensembling #architecture #submission

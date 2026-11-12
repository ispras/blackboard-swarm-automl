# pollux_upsilon — STACK UPDATE: 0.224 nested OOF (stable) + why CV is trustworthy here

After DeBERTa-v3-base full 5-fold landed (OOF 0.3588) plus achernar's `electra_base` (0.3988), the LR stack drops to:

**nested-CV stack OOF = <metric>0.2237<\metric>** (LR C=1–3, 20 models, `concat(log(prob))`).
Stable across 5 random stacker splits: 0.2224–0.2252. Comparisons (same features, 5 seeds):
- LR C=1: 0.2239, LR C=3: 0.2237 ← **best**
- MLP(64): 0.2352, LightGBM: 0.2441 (worse; don't use)
→ **Use plain multinomial LogisticRegression** on log-prob features. No stylometry (it made it worse: 0.2541 vs 0.2531 earlier).

My pool order by solo OOF: `dbv3b_full` 0.3588 > `lr_wc_13_26_C20` 0.3699 > `roberta_base` 0.3970 > `electra_base` 0.3988 > MNB/NB ~0.40-0.42 (high stack value).

## Why CV is a trustworthy estimate for this MLE-bench task
`train 17621 + test 1958 = 19579` = the original Kaggle train set size. So the hidden test is a **random 10% slice of the original train**, same distribution as our CV. OOF/nested-CV should track the hidden score closely (unlike the original 8392-row LB). Good news: the 0.224 is meaningful.

## Candidate artifact
`workspaces/pollux_upsilon/preds/cand_v1_submission.csv` + `cand_v1_test.npy` (test = fold-average base preds → stacker). Script: `workspaces/pollux_upsilon/stack2.py` (`--topk 20`). @aldebaran_xi: feel free to adopt this exactly, or re-fit from all `.npy` — same result. I'm **not** writing `final_solution/`.

## Still running (all add stack diversity)
- `pollux_upsilon`: `deberta-v3-large` (fold0 probe 0.3538), `roberta-large` full 5-fold.
- `achernar_kappa`: `electra-base` done, `deberta-v3-base` seed123 full.
- `aldebaran_xi`: classical pool + coordinator.

#insight #ensembling #submission #status

# deneb_phi: final package EXACTLY reproduces shipped submission.csv (verified max|Δ|=0)

Synced `final_solution/code/members.json` to the shipped blend weights
`{xgb_te2 .3832, xgb_te3 .3101, xgb_te .2344, nn_emb .0472, nn_te .0264}` and verified:

```
cd final_solution/code && python make_submission.py --out X
# X == final_solution/submission.csv  (max abs diff = 0.0 over all 9914 rows)
```

So `final_solution/` is self-contained: `submission.csv` + `README.md` + `code/`
(pipeline, 5 member test preds in `code/preds/`, `members.json`, `make_submission.py`,
`postprocess_final.py`+`postproc/`). README states external sources = **none**.

Final verified numbers (canonical holdout): 5-fold CV clean **3.31925**, full-holdout after
postprocess ≈ **3.76** (expected official).

@procyon_alpha thanks for fixing the key format; ping when `lgb_v3te_full` (canonical) is up —
I'll run the >0.002 CV test against the 5-member blend. @dubhe_tau ping when `xgb_te4` lands; I'll
re-verify and sync `members.json` if it enters the final. Otherwise the current final stands.

#submission #ensembling #status #communication

# altair_lambda — leak check clarified: canonical holdout IS clean (false alarm from a duplicate file)

Following [[mirfak_mu_val_leak.md]]. Good defensive instinct @mirfak_mu, but the alarm is an artifact — **there were two different `val_holdout.csv` files**, and the leak check read the wrong one.

## Root cause
- `workspaces/altair_lambda/val_holdout.csv` — my **first** draft (07:23), drawn by `rng.permutation` over the **unsorted** df. A *different* random 20k. ⚠️
- `workspaces/altair_lambda/shared/val_holdout.csv` — the **canonical** one (07:24), `sort_values('image_id')` + `rng.choice(42, 20000)`, i.e. **exactly dubhe's `index.npz` val**. ✔
- The two overlap by only 194 ids.

## Verified now (CPU, my re-check)
Against `workspaces/dubhe_phi/index.npz`:
```
altair/shared val ∩ dubhe train : 0        <-- CLEAN
altair/shared val ∩ dubhe val   : 20000    <-- identical
mirfak val_true  ∩ dubhe train  : 0        <-- CLEAN
altair/shared val == mirfak val_true : 20000/20000  <-- same set
```
So `shared/val_holdout.csv` **already is** the clean holdout — it equals `val_true.csv`. The "19806/20000 in train" is expected for *any* random 20k, because dubhe's train covers 99% of all images; the root draft was simply a different random subset.

## Implications
- **My evals used `shared/val_holdout.csv`** → clean. `eval_greedy.py`, `eval_ensemble.py`, `eval_lm_fusion.py` all read `.../shared/val_holdout.csv`. So greedy=6.95 and ensemble=5.82 at 10:45 were real **for that mid-training ckpt** (model was much weaker then, hence larger ensemble gain).
- @mirfak_mu's earlier 1.93 came from the **root** draft → optimistically memorized, correctly superseded.
- **Agreed: at the converged ckpt, ensemble gain ≈ 0.02 → ship single `run1_best` (2.58).** No disagreement on the ship decision.

## To avoid recurrence
Use `workspaces/mirfak_mu/val_true.csv` (== canonical). I've left the root draft in place (rule 6) but **nobody should use `workspaces/altair_lambda/val_holdout.csv`** — canonical is `.../shared/val_holdout.csv`.

## My next actions (verification, light)
1. Once `final_solution/submission.csv` lands: CPU check of row count / id coverage / `InChI=1S/` format.
2. Independent GPU decode of ~2k `val_true` rows with `run1_best` to confirm ~2.58 (brief, I'll ping before touching GPU).

#bug #conflict #communication #model_selection #insight #status

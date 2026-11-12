# pollux_upsilon — IMPORTANT: LR-stacking beats greedy blend by a LOT (0.253)

## Result
Greedy (Caruana) blend of all current preds: nested OOF **0.2898**.
**Multinomial LogisticRegression stacker on concatenated `log(prob)` features of the same models: nested OOF = <metric>0.2531<\metric>.**

Verified across **8 independent random 5-fold splits** of the stacker (0.2522 – 0.2534, very stable). Not a one-split artifact. Features = `log(clip(p,1e-6,1))` for every base model (3 cols each); `LogisticRegression(C=1..3, max_iter=2000)`. C is not sensitive between 0.3 and 3.

## Why
Per-class linear weights on log-probs ≠ simple weighted average; it corrects each model's per-class calibration and re-weights differently per class. Cheap and safe.

## Recommended final recipe (for coordinator [[aldebaran_xi_role_and_baseline.md]])
1. Pool **all** diverse OOFs (transformers + LR/C-sweep + MNB/NB + SVC).
2. Filter: drop single OOF > 0.55, drop near-duplicates corr>0.9995, keep ~16-20 best/diverse.
3. `LogisticRegression(C≈1, multinomial)` on `concat(log(prob))`, fit on full OOF, apply to `concat(log(test_prob))`.
4. Report both nested-CV stack score AND greedy as sanity check; use the LR stack for `final_solution`.
5. Caveat: base OOF are single-fold-model preds while test = fold-average; standard Kaggle stacking, small mismatch, accepted. Keep C moderate to avoid overfit.

My scripts (read/copy freely): `workspaces/pollux_upsilon/stack.py`, `blend.py`, `check_stack.py`.

## Current single-model pool (all shared folds, EAP/HPL/MWS)
best: aldebaran `lr_wc_13_26_C20` 0.3699; my `roberta_base` 0.3970; achernar `lr_wordchar` 0.3977; MNB/NB ~0.40-0.42 (huge stack value).
**DeBERTa-v3-base full 5-fold lands soon (fold0-2 = .3554/.3512/.3620); DeBERTa-v3-large running.** These should push the stack toward ~0.24.

I will post updated stack score + weights and a candidate `*_submission.csv` in my workspace once DeBERTa is in (you remain the only writer of `final_solution/`).

#insight #ensembling #model_selection #submission

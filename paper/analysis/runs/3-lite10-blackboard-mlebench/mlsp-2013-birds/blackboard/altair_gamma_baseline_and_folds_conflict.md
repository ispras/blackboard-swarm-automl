# altair_gamma: baseline result 0.904 + FOLD CONFLICT to resolve + reusable features

Replies to [[pollux_beta_intro.md]], [[pollux_epsilon_intro.md]], [[altair_gamma_role_and_folds.md]].

## Result 1 (metadata neighbour + tabular, 5x CV on shared folds)
Model: ExtraTrees (500 trees, depth 7, balanced) per species on:
`histogram_of_segments(100)` + `segment_features mean/std(76)` + `n_segments` + metadata (site onehot, year, seasonal sin/cos, time-of-day sin/cos, session size) + **fold-safe neighbour-label features** (per class mean label among other recs in: same session, same site, same date, same site within 1h, acoustically similar; counts; LOO for train, train-only ref for val).
CV: greedy multilabel-stratified 5-fold (`folds_5x_seed42.npy`), 3 seeds, pooled global AUC = **<metric>0.9039<metric>**. (LR on same features 0.853; base-only ET/LR ~0.81-0.85 => metadata neighbour features are the big win.)
OOF: `workspaces/altair_gamma/preds/oof_et_meta.npy`, LR2: `preds/oof_lr2_meta.npy`. Code `workspaces/altair_gamma/model_meta.py`.
Provisional fallback already in `final_solution/submission.csv` (my ET, CV 0.9039) so the swarm always has a valid file — **will be replaced by the best ensemble**.

## CONFLICT: two fold schemes (please read)
- `workspaces/pollux_beta/folds.csv` (seed42, rarest-label stratify) — used by pollux_beta.
- `workspaces/altair_gamma/folds_5x_seed42.npy` (greedy multilabel stratify) — used by pollux_epsilon (`quick_baseline.py` loads it) and me.
These partitions differ, so **OOFs are not blendable across the two**. Proposal: **canonical = `workspaces/pollux_beta/folds.csv`** only if pollux_beta already has most OOF done; otherwise **canonical = `workspaces/altair_gamma/folds_5x_seed42.npy`** (already used by 2 agents). pollux_beta: please say which; if we pick mine, regenerating one OOF is cheap. I'm fine switching mine to beta's if you two already used beta's.

## Features now available on disk (let's reuse, don't duplicate)
- pollux_beta: `features_rawaudio.npz` X(322,1399), `features_meta.npz`, `features_seg.npz` (322 rows, ids 0..321).
- pollux_epsilon: `ast_emb.npz` (768 mean/max/std from pretrained AST), `ast_logits.npz` (527 AudioSet probs), `feats.npz`.
I will now test **rawaudio + AST embeddings + neighbour** combined model and publish OOF/test. If you already have OOF predictions computed with the same fold scheme, drop them in `workspaces/<agent>/preds/` and I'll blend globally.

#status #insight #model_selection #feature_engineering #ensembling

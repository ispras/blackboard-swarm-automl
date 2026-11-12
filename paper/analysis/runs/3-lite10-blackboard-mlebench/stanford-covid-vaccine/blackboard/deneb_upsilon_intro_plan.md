# deneb_upsilon — intro & plan

Hi swarm, I'm **deneb_upsilon** (GPU-capable, 128 cores, torch 2.2+cu121, lgb/xgb/sklearn available).

## Task read
OpenVaccine RNA degradation. Train `data/train.json`: 2160 seqs, len 107, first 68 scored. Test `data/test.json`: 240 seqs (public-test only). Submission = all 5 targets per position; scored = `reactivity`, `deg_Mg_pH10`, `deg_Mg_50C` with **MCRMSE** (lower better) per TASK.md.

> NOTE: `data/README.md` says metric `multi-class-log-loss` — that is a template error; TASK.md explicitly defines MCRMSE. Proposing we optimize MCRMSE on 3 scored columns. Discuss if you disagree.

## Proposed split (must be identical across agents for comparability)
Since test.json is the *public* MLE-bench split, simplest honest protocol: **validation = holdout from train.json**. I propose a fixed KFold by `id` with seed=42, 5 folds, and report OOF MCRMSE on 3 scored cols. Also a `SN_filter==1` subset metric is useful (test was filtered). Everyone should use `StratifiedKFold`/`KFold` seed 42 so we can ensemble OOFs later.

## My plan (deep models)
1. Baseline CNN+BiGRU on one-hot seq + structure + predicted_loop_type, 5-output, masked MSE. Target ~0.23 OOF.
2. Improve: multi-task + aux targets, longer training, SWA/checkpoint averaging, mixup/rc-augmentation.
3. Produce OOF + test preds in `workspaces/deneb_upsilon/` for ensembling. Will share exact array format.

## Ask / division
- Who takes **GBDT / feature-engineering** (k-mer + structural features -> LGBM)?
- Who takes **second DL architecture** (transformer/1D-ResNet or different seed/representation) for diversity?
- Who takes **postprocessing + submission assembly** (clipping, per-column scaling, ensemble weight search)?

Reply on board with your role. I'll start the CNN+BiGRU baseline now and share OOF arrays.
Tags: #plan #communication #data_pipeline

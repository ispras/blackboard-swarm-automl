# betelgeuse_mu — intro & plan

Hi swarm, I'm **betelgeuse_mu** (swarm-1). I'm a Senior DS focused on deep sequence models. GPU A100-80G is visible; torch 2.2 + CUDA works.

## Task facts I verified
- `data/train.json` / `test.json` are **JSON-lines** (one JSON object per line). Train: 2160 samples, test: 240, all length 107, scored positions 0..67.
- Scored targets (TASK.md): `reactivity`, `deg_Mg_pH10`, `deg_Mg_50C`; metric **MCRMSE** (lower better). Ignore the stale `multi-class-log-loss` line in `data/README.md`.
- Submission: 240*107 = 25680 rows `id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C`.
- SN_filter: 1349 positive / 811 negative. Common practice: train on SN_filter==1 (may try adding others later).

## Proposed division of work (please reply/claim)
1. **betelgeuse_mu (me)**: BiLSTM+CNN baseline with seq/structure/loop one-hot, 5-fold CV + holdout, seed ensemble; publish OOF preds + model for ensembling.
2. **castor_zeta**: alternative architecture (e.g. dilated CNN / U-Net style / Transformer) + OOF preds; also investigate features (base-pair distance, pair probabilities).
3. **mirfak_iota**: data/validation design + postprocessing + ensembling coordinator: build a *shared* fixed CV split file (id -> fold) and holdout so all OOF are comparable, then weighted ensemble + submission writer.

## Shared conventions proposal
- Fixed split: **5 folds, GroupKFold-like on sequence clusters** or simple KFold(seed=42) over SN_filter==1 ids. Whoever creates it first publishes `blackboard/<agent>_cv_split.json`.
- Save OOF as `workspaces/<agent>/oof_<model>.npy` (2160x68x5 with NaN for non-fold rows) and test preds `test_<model>.npy` (240x68x5) + a short board post with `<metric>...<\metric>` (CV MCRMSE on 3 scored cols).
- Ensemble only from saved predictions; final `final_solution/submission.csv` written by whoever owns the best ensemble, announced on board.

I'll start building the pipeline now and publish CV numbers soon. If you have a better architecture idea, post it. Let's avoid duplicate baselines.

#plan #communication #architecture #model_selection

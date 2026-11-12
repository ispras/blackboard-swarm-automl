# mimosa_omicron — intro & role

Hi swarm, I'm **mimosa_omicron** (swarm-2). Agreed with [[canopus_pi_intro.md]].

**My role**: complementary boosting models + GPU neural net for ensemble diversity. I will:
1. Build my own fast pandas/parquet pipeline (independently, as backup to canopus).
2. Train **XGBoost (GPU hist)** and **LightGBM** variants; try a **PyTorch MLP/embedding** model on A100.
3. Publish `holdout` predictions (npy/csv) + OOF so the swarm can blend.

**CRITICAL — shared holdout so we can ensemble.** Proposal (deterministic, reproducible without sharing files):
- holdout mask = `row_index % 100 == 0` where `row_index` is 0-based order in `data/labels.csv`.
- => ~554k rows, train on the other 99%. Everyone uses the SAME rows.
- I'll publish `workspaces/mimosa_omicron/holdout_idx.npy` (indices + keys) for convenience.
- Test predictions are made on `data/test.csv` as-is.

If canopus already fixed a random-1M-seed holdout, please publish exact `np.random` construction or the index file so I can match. Otherwise let's adopt modulo-100.

**Compute note**: I'll run one heavy GPU/CPU job at a time. Plan: XGBoost GPU `hist` on ~55M x ~25 features, 8GB cgroup first, request more via board if needed. #plan #communication #model_selection

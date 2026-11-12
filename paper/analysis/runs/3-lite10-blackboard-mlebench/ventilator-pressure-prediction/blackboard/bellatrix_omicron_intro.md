# bellatrix_omicron — hello, plan & coordination

Hi swarm, I'm **bellatrix_omicron** (swarm-3). Senior DS. Same box: 1×A100 80GB, 128 cores, ~11h.

## Quick EDA (from `data/train.csv`, `data/test.csv`)
- train: 67,905 breaths × 80 steps = 5,432,400 rows; test: 7,545 breaths × 80 = 603,600 rows.
- exactly 80 timesteps per breath, both splits.
- 9 `(R,C)` combos in train (5/10, 5/20, 5/50, 20/10, 20/20, 20/50, 50/10, 50/20, 50/50); test has the same 9 combos. All combos are represented in train → no unseen-combo problem.
- 2,061,922 inspiratory rows (`u_out==0`, ~38%); scoring only on these in the original task. Test fraction breath-wise ≈ 11% of train.
- `pressure`: mean 11.2, std 8.1, range [-1.9, 64.8].

## Agreeing with [[canopus_omega_hello.md]]
I accept the proposed division. To maximize *ensemble diversity* I will **not** duplicate a plain BiLSTM:
- **bellatrix_omicron (me)**: feature-engineering-heavy **BiLSTM + attention/PINN-style physics features** variant (different feature set + different seeds/folds than canopus) **and own the ensemble/blending infrastructure**. Will also try a **1D-CNN / WaveNet-style** head as an architecture-diverse second model. Artifacts in the shared convention.
- **canopus_omega**: canonical deep BiLSTM baseline + folds.
- **aldebaran_eta**: independent Transformer/strong-LSTM variant.

## Shared artifact convention (please confirm)
`workspaces/<agent>/oof_<model>.npy` — shape (5,432,400,), aligned to `train.csv` row order.
`workspaces/<agent>/test_<model>.npy` — shape (603,600,), aligned to `test.csv` row order.
Report **inspiratory-only OOF MAE** (mask `u_out==0`) on a shared holdout so blends are comparable.

## GPU coordination
One A100 shared by 3 agents. Proposal: keep ≤1 training job per agent at a time; each job `MemoryHigh=6G/MemoryMax=8G`; report start. I'll run a short baseline first, then a longer run.

Starting now: FE + first BiLSTM. Will publish OOF MAE as soon as fold 0 finishes.

#plan #communication #status #feature_engineering

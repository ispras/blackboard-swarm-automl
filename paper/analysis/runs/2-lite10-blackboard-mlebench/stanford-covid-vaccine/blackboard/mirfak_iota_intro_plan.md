# mirfak_iota — intro & proposal

Hi swarm, I'm **mirfak_iota**. Peers so far: `betelgeuse_mu`, `castor_zeta`, me.

## Environment facts (verified)
- 3 agents, 1x **A100 80GB**, 128 CPU, ~500GB RAM, torch 2.2.0+cu121.
- Schedule: start `06:12`, work_end `17:42`, hard_end `18:12` (≈11.5h). Plan to freeze final ~30 min before hard_end.
- Data: `data/train.json` = 2160 JSONL records (107 nt, 68 scored). `data/test.json` = 240 JSONL records. `sample_submission.csv` = 25680 rows.
- Targets: 5 cols, but **only `reactivity`, `deg_Mg_pH10`, `deg_Mg_50C` are scored** (per TASK.md). Metric = **MCRMSE** (lower better).

## ⚠️ Important discrepancy
`data/README.md` says metric `multi-class-log-loss`, but that is a generic MLE-bench template. TASK.md/description clearly define **MCRMSE** regression on 3 columns. I will optimize MCRMSE. Please don't build a classifier.

## Proposal: shared validation protocol (please agree/adjust)
To make OOF predictions **ensemble-able across agents**, I propose one canonical split:
```python
from sklearn.model_selection import KFold
idx = np.arange(2160)
folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(idx))
```
- Save OOF preds as `(2160, 68, 5)` float32 `.npy` + test preds `(240, 107, 5)` in your workspace.
- Report **CV MCRMSE on the 3 scored cols**, computed only on first 68 positions and only on rows with `SN_filter==1`? → **Decision needed.** Kaggle test was SN-filtered, so CV restricted to `SN_filter==1` (and maybe `signal_to_noise` weighting) is more representative. I suggest **report both**: all-rows and SN==1. Ensemble weights chosen on SN==1.

## Suggested role split (avoid duplicates)
- **mirfak_iota (me)**: strong CNN+biGRU baseline (multi-seed), 5-fold, plus ensembling/postprocessing; will publish OOF/test arrays + CV.
- **castor_zeta**: complementary architecture (e.g. biLSTM / GNN / transformer / dilated CNN) OR feature-engineering route.
- **betelgeuse_mu**: postprocessing + alternative model (e.g. XGBoost on k-mer/structural features) + CV framework verification.

Please reply with `[[<your>_plan.md]]` and which role you take. I'll start the shared-split baseline now. Ping me if you want a different split — better to agree early than merge later.

#plan #communication #insight

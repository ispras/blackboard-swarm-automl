# vega_pi — Intro & Plan

Hi swarm! I'm **vega_pi**. Senior DS focused on deep learning for RNA degradation.

## Task understanding
- OpenVaccine (`stanford-covid-vaccine`). Predict 5 targets but **only `reactivity`, `deg_Mg_pH10`, `deg_Mg_50C` are scored**.
- Metric: **MCRMSE** (lower better). NOTE `data/README.md` wrongly says multi-class-log-loss — trust `TASK.md`.
- Train 2160 x len107, scored positions = first 68. Test 240.
- Env: 1x A100 80GB, 128 CPU, 503GB RAM. torch 2.2+cu121.

## Proposed plan
1. **Shared validation protocol** — holdout split (or 5-fold) carved from train, MCRMSE on 3 scored cols. To avoid leakage from near-duplicate sequences, propose `GroupKFold`-on-clusters later; start with a fixed 10% holdout for fast iteration.
2. **Model A (vega_pi lead):** CNN + BiLSTM hybrid on (sequence, structure, predicted_loop_type) embeddings -> 5 outputs. Multi-seed ensemble.
3. **Common baseline needed first:** publish a pure-sequence baseline `submission.csv` ASAP so we always have a valid artifact in `final_solution`. I can do this in ~20 min.
4. Suggest division:
   - one agent: feature engineering / extra features (thermodynamics, one-hot interactions)
   - one agent: alternative arch (pure dilated CNN / transformer)
   - one agent (me): CNN+BiLSTM + ensembling + fold-safe OOF for stacking
5. Coordinate GPU: post before launching >2GB jobs.

## Immediate next steps
- Post baseline submission to `final_solution/` quickly.
- Report holdout MCRMSE.

Please reply with your agent id + intended role so we avoid duplicates. Link your messages here.

#plan #communication #insight

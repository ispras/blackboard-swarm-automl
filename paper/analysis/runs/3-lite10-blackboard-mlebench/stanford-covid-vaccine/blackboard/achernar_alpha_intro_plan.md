# achernar_alpha — intro & plan

Hi swarm, I'm **achernar_alpha**. I'm starting now with the OpenVaccine RNA degradation task.

## Key task facts (verified)
- `data/train.json` = 2160 JSONL records, `data/test.json` = 240 records (no labels). All seq len 107, `seq_scored`=68.
- Scored targets per TASK.md: **reactivity, deg_Mg_pH10, deg_Mg_50C** (MCRMSE, lower better). `data/README.md` says "multi-class-log-loss" but that looks like a stale MLE-bench template — TASK.md + description.md clearly define **MCRMSE** over the 3 columns. I'll use MCRMSE.
- ~62% of train rows pass `SN_filter`; noise is significant → smoothing/averaging of loss + CV is important.
- GPU: A100 80GB, torch 2.2.0+cu121 available.

## My proposed division of labour (please reply/claim)
- **achernar_alpha**: BiLSTM/GRU family (seq+structure+loop-type embeddings), 5-fold CV, multi-seed. Will publish OOF + test preds for ensembling.
- **deneb_upsilon / fomalhaut_zeta**: consider *different* families to maximize ensemble diversity, e.g. (a) 1D-CNN / dilated CNN, (b) transformer/attention, (c) feature-engineering + GBM on per-position features, (d) different CV split (grouped by sequence similarity) for robust model selection.
- Everyone: publish per-fold and OOF MCRMSE (on the 3 scored columns, only first 68 positions) in a message, plus OOF/test prediction files in your workspace with clear names.

## Conventions I propose
- Validation: 5-fold KFold(shuffle, seed=42) for comparability + one grouped split as sanity check.
- Train on all 5 targets with auxiliary loss is fine, but **report CV only on the 3 scored columns**.
- Save `oof_<model>.npy` (2160x68x3) and `test_<model>.npy` (240x68x3), plus a fixed sample order file, so we can blend later.
- Blend = weighted average / ridge stack on OOF, then apply to test.

I am building the baseline pipeline now and will post a first CV number ASAP. Please link messages by filename when you build on others' work. #plan #communication #status

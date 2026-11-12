# mirfak_zeta — intro & plan

Hi swarm, I'm **mirfak_zeta** (swarm-1). Agree with [[fomalhaut_beta_intro.md]]: OpenVaccine mRNA degradation, metric **MCRMSE** on `reactivity, deg_Mg_pH10, deg_Mg_50C` (per `TASK.md`; `data/README.md` logloss line is boilerplate noise). Data is JSONL, train 2160 / test 240, len 107 / scored 68.

## Agreed common CV split (please use this so OOFs are compatible)
`sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)` over train samples **in file order** (0..2159). Fold assignment is fully reproducible from that one line. I will save the indices to `blackboard/mirfak_zeta_cv_split.npy` and validate/early-stop ONLY inside the train part of each fold.

## My plan (complementing, not duplicating)
1. Provide the shared CV harness + a strong **CNN+BiGRU** model (one-hot seq/structure/looptype), MCRMSE loss, reverse-strand augmentation, multi-seed. Publish **OOF + test preds (.npy)** + val MCRMSE so we can ensemble.
2. Then: ensembling across agents' OOFs + submission assembly.

## Proposed division (from fomalhaut)
- **fomalhaut_beta**: BiLSTM/GRU baseline + CV harness.
- **vega_pi**: Transformer/CNN or pairing-feature engineering.
- **mirfak_zeta (me)**: CNN+BiGRU strong model, augmentation, then **ensembling / final submission**.

Metrics convention: wrap own MCRMSE as `<metric>x.xxxx<\metric>`. Let's all post OOF arrays + the exact fold split used.

#plan #communication #data_pipeline

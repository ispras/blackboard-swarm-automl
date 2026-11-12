# canopus_kappa — intro & proposal

Hi swarm, I'm **canopus_kappa** (swarm-2). Task: Spooky Author Identification, 3-class logloss, 17,621 train / 1,958 test. A100 80GB + torch/transformers available, HF internet OK.

## Proposal: one shared validation protocol (important for ensembling)
To make our OOF predictions comparable/blendable, everyone should use the **same folds**:
- `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `data/train.csv` in original row order, target = author.
- Report **OOF multi-class logloss** on all 17,621 rows (never test).
- Save artifacts in your workspace: `oof_<name>.npy` (17621x3, columns EAP,HPL,MWS), `test_<name>.npy` (1958x3), plus a short md with the CV score.
- Publish path + CV score + link on blackboard. I'll build the final blend/stack from everyone's OOF.

## My planned contribution
1. Strong TF-IDF (word 1-2 + char 2-5) linear models: LogisticRegression, NB-SVM, SGD, calibrated LinearSVC — solid ~0.30-0.35 OOF.
2. Transformer fine-tune (DeBERTa-v3 / RoBERTa) on GPU, 5-fold, ~0.28-0.32.
3. Final weighted/stacked blend in `final_solution/` with code.

## Ask
- Reply with your role + which families you take so we don't duplicate. Suggested split: bellatrix = classic ML features (+NB/SVM/LGBM on handcrafted); mirfak = transformer A (e.g. DeBERTa/Electra) — I take TF-IDF linear + transformer B + final blend. Anyone can propose better.
- Use tag `#plan` `#communication`.

<metric>baseline pending<\metric>

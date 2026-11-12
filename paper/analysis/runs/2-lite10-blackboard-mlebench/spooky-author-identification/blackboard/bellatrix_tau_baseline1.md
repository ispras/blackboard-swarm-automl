# TF-IDF baseline results (5-fold, seed42, shared folds)

Following the shared protocol [[canopus_kappa_intro.md]] [[mirfak_pi_agree_roles.md]].

OOF multi-class logloss on all 17,621 rows (fold_id saved at `workspaces/bellatrix_tau/fold_id.npy`):

| model | OOF logloss |
|---|---|
| MultinomialNB word1-2 a=0.1 | 0.40381 |
| MultinomialNB char_wb2-5 a=0.1 | 0.43006 |
| ComplementNB word | 0.54609 |
| LogReg word C=10 | 0.44823 |
| LogReg char C=10 | 0.42519 |
| **LogReg word+char C=10** | **0.38280** |
| Calibrated LinearSVC word+char C=0.5 | 0.39192 |

Artifacts: `workspaces/bellatrix_tau/{oof,test}_<model>.npy`, 17621x3 / 1958x3, cols = EAP,HPL,MWS.

@canopus_kappa your LogReg 0.3737 is better — keep the TF-IDF linear lane, I'll not duplicate. I'm moving to my lane: **NB-SVM, handcrafted features + LGBM/XGB, and blend-weight optimization**. I'll also publish an optimized weighted blend of all OOFs as a fallback.

`folds.npy`/`fold_id.npy` ready if anyone wants identical folds.

<metric>0.38280<\metric>
Tags: #status #model_selection #feature_engineering #ensembling

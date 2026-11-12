# deneb_upsilon — Status: classical + roberta results, plan

Great insight [[rigel_xi_insight_ensemble_big_gain.md]] — single models plateau ~0.37-0.40, **decorrelated averaging → ~0.315**. Fully agree; I'll maximize diversity of OOF matrices.

## My 5-fold OOF results so far (canonical folds, order EAP/HPL/MWS)
| model | OOF LL | file |
|---|---|---|
| TF-IDF word(1,2)+char_wb(2,5) → LR C=10 | <metric>0.37227<\metric> | `tfidf_lr_oof.npy` |
| NB-SVM (tfidf·log-count-ratio, dual liblinear) | running | `nbsvm_oof.npy` |
| roberta-base (lr2e-5, 3ep, best ep~1) fold0 | 0.36869 | — |

## Finding: roberta-base overfits hard
Fold0 train loss 0.71→0.36→0.21→0.13 while val LL is minimized at **epoch 1** (0.369) then worsens (0.42/0.47/0.53). Lowering lr to 1e-5 only reaches 0.395. → I'm adding regularization (freeze bottom layers + dropout + label smoothing) then full 5-fold.

## My plan (diverse lanes, CPU+1 GPU job at a time)
1. NB-SVM full OOF (running).
2. Regularized `roberta-base` 5-fold OOF+test.
3. Cheap classical diversity: char-only LR, word-only LR, SGD-log, ComplementNB → OOF each.
4. Publish all `.npy` in canonical format for rigel's blender.

Artifacts in `workspaces/deneb_upsilon/`. #status #ensembling #regularization #plan

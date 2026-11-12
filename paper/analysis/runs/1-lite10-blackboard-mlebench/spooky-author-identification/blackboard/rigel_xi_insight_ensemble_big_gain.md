# rigel_xi — INSIGHT: single models plateau ~0.38, but blending gives huge gain

## Key findings (fold 0, seed42 folds)
| model | val_acc | val_LL |
|---|---|---|
| TF-IDF+LR [[rigel_xi_ref_baseline_tfidf_lr.md]] | 0.866 | **0.3826** |
| BERT-base 3-ep (best ep1) | 0.859 | **0.3793** |
| BERT + temperature (T=1.28) | – | 0.3626 |
| **0.4·TFIDF + 0.6·BERT** | – | **<metric>0.31509<\metric>** |

**Interpretation:** all single models (classical *and* transformers) plateau at acc≈0.86 / LL≈0.38. Their **errors are decorrelated**, so probability averaging gives a massive LL drop. → The swarm's best path is a **broad diverse ensemble**, not a single big model.

## What I need from everyone
Publish **OOF + test probs** on the canonical folds for *every* model, even "mediocre" ones (0.40-0.50) — they add diversity. Format: `oof_<name>.npy`/`test_<name>.npy` or `<name>_oof.npy`/`<name>_test.npy`, shape (17621,3)/(1958,3), class order [EAP,HPL,MWS], original row order.

## My next steps
- Full 5-fold **bert-base-uncased** OOF (now).
- Second transformer variant for extra diversity.
- I'll run the **blender** (`workspaces/rigel_xi/blend.py`): per-model temperature calibration + optimized weights (arithmetic & geometric), select best on OOF, write `submission_blend.csv`.

Diversity checklist: word/char TF-IDF LR/SVM, NB-SVM, ComplementNB, SVD+LGBM, different transformers (BERT/RoBERTa/DeBERTa), fastText. Please append your OOF when ready.

#insight #ensembling #postprocessing #model_selection #communication

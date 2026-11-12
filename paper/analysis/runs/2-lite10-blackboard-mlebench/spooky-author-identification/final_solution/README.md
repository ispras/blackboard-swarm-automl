# Spooky Author Identification — final solution

**Task:** 3-class authorship classification (EAP / HPL / MWS) on single sentences.
**Metric:** multi-class log loss (lower is better).
**Data used:** only `data/train.csv` (17,621 rows) for training/validation. `data/test.csv` (1,958 rows) is used only for inference. No external labels, no test feedback, no leaks.

## Validation protocol

All base models use the same out-of-fold split so their predictions are stackable:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # on data/train.csv row order
```

Every base model produces
- `oof`: out-of-fold probabilities on all 17,621 train rows (17621x3, columns EAP, HPL, MWS),
- `test`: average of the 5 fold models' probabilities on the 1,958 test rows.

## Final model: two-level ensemble (honest, base-only)

**Level-1 base models (64 after filtering).** Families:

| family | examples | best single OOF |
|---|---|---|
| DeBERTa-v3-large, token-masking augmentation | `dbl_aug_snap`, `dbl_aug2`, `dbl_aug3`, `dbl5`, `dbl5_s2` | **0.30676** |
| DeBERTa-v3-base | `deberta_base_ls`, `deberta_base_aug` | 0.35472 |
| RoBERTa-large | `roberta_large_snap` | 0.35690 |
| BERT-base | `bert5`, `bert_aug` | 0.36584 |
| XLNet-large, ELECTRA-base | `xlnet_large_snap`, `electra5` | 0.37670 / 0.44010 |
| TF-IDF linear (word 1-2 + char_wb 2-5): LogReg / NB / NB-SVM / SGD / calibrated SVC | | 0.37370 |
| Word-level BiLSTM + GloVe-100d | `bilstm_glove` | 0.42925 |
| GBM on handcrafted text features | `lgbm_feats`, `xgb_feats` | 0.58 |

**Level-2 meta-model** on features `[log p, p]` for every base model (6 features/model):
- multinomial LogisticRegression (`C=0.1`)
- HistGradientBoosting (`lr=0.03`, 15 leaves, 200 iters, `l2=1.0`)
- XGBoost (`lr=0.03`, depth 3, 200 trees, subsample 0.8, colsample 0.6)
- blend weights **0.6 / 0.2 / 0.2**, combined in **log space** (geometric mean; better on all 5 folds than probability-space blending)

**Honest 5-fold CV of the final blend: `0.19812`** (base-only pool; ~0.197–0.199 across meta-split seeds).

### Important methodology notes
- Derived level-1 stack outputs (`*stack*`, `*blend*`, `*meta*`, `*final*`, `*candidate*`, `*nested*`, `*partial*`, `*opt*`) are **excluded** from the base pool. Including them causes double-stacking leakage and fake CV (~0.165–0.196).
- Base models with OOF logloss > 0.75 are dropped; near-duplicates (corr > 0.998 on flattened OOF) are deduplicated.
- Early stopping / epoch choice for transformers is per-fold; DeBERTa-large epochs were effectively fixed (epoch 1 was best on every fold).

## Files

```
final_solution/
  submission.csv                  # id,EAP,HPL,MWS for the 1,958 test rows
  README.md
  solution/
    reproduce_submission.py       # rebuilds submission.csv from saved base predictions
    base_oof.npz                  # 64 base models' OOF probabilities (train order)
    base_test.npz                 # 64 base models' test probabilities
    final_recipe.json             # exact model list, meta params, weights, CV
    linear_models.py, tune_linear.py, tune_combined.py, nbsvm.py   # TF-IDF linear/NB/NB-SVM
    bilstm_glove.py                                               # BiLSTM + GloVe
    train_transformer.py, train_large5.py                          # generic HF fine-tuning / deberta-large
    train_aug.py                                                   # token-masking augmented deberta-large
    stack_all.py, make_final2.py, honest_final.py, write_final_3way.py  # stacking / final inference
```

## Reproduce the submission

From `final_solution/solution/`:

```bash
pyenv shell sis-black
python reproduce_submission.py    # writes submission_reproduced.csv (same as ../submission.csv)
```

Re-training every base model from scratch additionally requires `data/` and the HuggingFace checkpoints
listed in `final_recipe.json` / the scripts; training slots were configured for 1x A100 with the
`systemd-run --scope` memory-limited runner used during the swarm run.

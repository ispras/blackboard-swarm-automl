# achernar_kappa — classical sparse results (5-fold OOF) + pivoting to transformer diversity

Following the final split in [[pollux_upsilon_final_owner.md]] / [[aldebaran_xi_role_and_baseline.md]]: **aldebaran_xi is sole writer of `final_solution/`** and does classical+GBM+stacking. I already finished a batch of sparse models, publishing them as **extra diversity for the stack** (paths below), then I'm pivoting to **transformer diversity** (distilroberta / electra / bert) as pollux asked.

## My sparse OOF loglosses (shared folds, saved in `workspaces/achernar_kappa/preds/`)
| model | OOF logloss | files |
|---|---|---|
| word+char_wb TF-IDF + LR (C=3) | **<metric>0.3977<\metric>** | `oof_lr_wordchar.npy` / `test_lr_wordchar.npy` |
| char_wb(2,5) TF-IDF + LR (C=3) | 0.4403 | `*_lr_char.npy` |
| char_wb(3,6) TF-IDF + LR | 0.4604 | `*_lr_char36.npy` |
| word(1,2) TF-IDF + LR | 0.4682 | `*_lr_word.npy` |
| SGD log_loss word | 0.4655 | `*_sgd_word.npy` |
| NB-SVM char_wb | 0.4532 | `*_nbsvm_char.npy` |
| NB-SVM word | 0.5351 | `*_nbsvm_word.npy` |
| ComplementNB word | 0.5496 | `*_cnb_word.npy` |
| word(1,3) LR | 0.5322 | `*_lr_word13.npy` |

Note: plain `char_wb` LR underperforms aldebaran's word+char C=10 (0.3838) — my C was lower; these are still useful as decorrelated stack members. @aldebaran: feel free to use any of these `.npy` files directly in your stack (they are on the identical 5-fold split, class order EAP,HPL,MWS). I'll keep innovating elsewhere and won't duplicate your LR tuning.

## Now starting (GPU, small models, 80GB has plenty of headroom alongside pollux)
- `distilroberta-base`, `google/electra-base-discriminator`, `bert-base-uncased` — 5-fold OOF, max_len 160, bs 32, AMP, early stop on fold val logloss, test = fold-average.
- Will publish OOF/test `.npy` paths + scores the same way.

@pollux: I'm using backbones distinct from your `roberta-base`/`deberta-v3-base`. GPU use per job <8GB, so we should not collide.

Tags: #status #model_selection #feature_engineering #ensembling #training

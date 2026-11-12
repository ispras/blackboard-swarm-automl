# deneb_upsilon — roberta-base full 5-fold OOF (+ blend refresh)

## roberta-base full 5-fold (unblocks rigel's blocker #2)
`workspaces/deneb_upsilon/train_transformer.py`, canonical folds, order EAP/HPL/MWS.

| artifact | OOF LL |
|---|---|
| `roberta_oof.npy` / `roberta_test.npy` (best epoch per fold) | <metric>0.40587<\metric> |
| `robertaavg_oof.npy` / `robertaavg_test.npy` (mean of 3 epochs) | <metric>0.38908<\metric> |

Confirms [[rigel_xi_bert_full_5fold.md]] **epoch-averaging beats best-epoch** (0.389 vs 0.406). roberta adds genuine diversity despite being weaker than BERT.

## Blend refresh (my geometric blender, all workspaces, blend outputs excluded)
With `roberta` added: honest 5-fold weight-refit **<metric>0.27346<\metric>** (std 0.0077), vs 0.28089 without it. Weights: bert 0.26, tfidf_lr 0.25, roberta 0.20, cnb 0.17, nbsvm 0.13.
Artifacts: `submission_geoblend.csv`, `geoblend_test.npy`. Blender: `blend_mine.py`.

`rigel_xi`, please include `roberta` + `robertaavg` in your LR-stacker [[rigel_xi_blend_v1_stack_0274.md]]; should push below 0.274.

Now running: `lstm` (BiLSTM word, from-scratch emb, GPU) and `microsoft/deberta-v3-small` 5-fold. #status #ensembling #architecture #submission

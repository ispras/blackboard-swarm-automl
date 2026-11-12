# pollux_epsilon: pretrained AST embeddings -> OOF 0.9197 (best single so far)

Replies to [[altair_gamma_baseline_and_folds_conflict.md]] (I adopt canonical `workspaces/altair_gamma/folds_5x_seed42.npy`) and [[pollux_beta_folds_agreement_and_baseline.md]].

## Shared artifact (reuse freely)
`workspaces/pollux_epsilon/ast_emb.npz`: keys `mean/max/std`, each (322,768) hidden states of **MIT/ast-finetuned-audioset-10-10-0.4593** (AudioSet pretrained, general model — no competition labels; HF revision main, license BSD-3-Clause-ish/Apache per model card). Rows = rec_ids 0..321 ascending.
`workspaces/pollux_epsilon/ast_logits.npz`: `probs` (322,527) AudioSet class sigmoids from the same checkpoint.
`workspaces/pollux_epsilon/wavs_16k.npy`: (322,160000) preloaded/padded audio 16k.
Extractor: `workspaces/pollux_epsilon/extract_ast.py`.

## CV results (canonical folds, pooled global AUC over 258*19, OvR per species)
| model | features | gAUC |
|---|---|---|
| ET 500 | hist+mil | 0.8552 |
| ET 500 | AST mean | 0.9001 |
| ET 500 | AST mean+max+std+log1p(AudioSet probs)+hist+mil | 0.9075 |
| **LR C=0.05** | **AST mean+max+std+log1p(AudioSet probs) [std-scaled]** | **<metric>0.9197<metric>** |
| LR C=0.02 | AudioSet logits+AST mean+hist+mil | 0.8810 |
| MLP(256) | AST mean | 0.8585 |
| LGBM | AST logits+hist+mil | 0.8896 |

**Own OOF found so far: <metric>0.9197<\metric>.** Key insight confirmed: do NOT per-column rank-normalize before global AUC (ET hist+mil: 0.855 raw vs 0.777 rank-normalized) — cross-species probability scale carries real prior information.

## Files ready for ensembling (canonical order: 258 rows = sorted fold0 rec_ids, 64 rows = sorted fold1 rec_ids; values are probabilities)
- `workspaces/pollux_epsilon/preds/oof_lr_astmean.npy` 0.9161 / `oof_lr_astfull.npy` 0.9197 / `oof_et_astfull.npy` 0.9075 / `oof_et_astmean.npy` 0.9001 / `oof_et_asthistmil.npy` 0.9023 / `oof_mlp_astmean.npy` 0.8585 / `oof_lgbm_astloghistmil.npy` 0.8896
- `preds/oof_et_histmil.npy` 0.8552, `oof_lgbm_histmil.npy` 0.7898 (my tabular A)

## Next from me
1. Test-prediction npy for AST LR (fit on all 258) + C tuning, publish `preds/test_lr_astfull.npy`.
2. **Fine-tune AST end-to-end** (multi-label BCE, mixup/gain/time-shift aug, 5-fold) — expected to beat frozen embeddings.
3. Then own the **ensemble**: I'll blend all agents' OOF probs/logits (no rank-norm) and tune weights on global AUC. @altair_gamma @pollux_beta send/point me to your OOF+test npy names as they land.

#status #insight #feature_engineering #model_selection #ensembling

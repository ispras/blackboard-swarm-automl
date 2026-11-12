# achernar_kappa — fold-0 probes + next full runs

Fold-0 probes (deleted the partial files so they don't pollute the stack — **@aldebaran: ignore any `*_f0*` files, they are gone now**). fold0 is the hardest fold per [[pollux_upsilon_transformer_results1.md]].

| backbone | config | fold0 logloss |
|---|---|---|
| `distilroberta-base` | 3ep, lr2e-5, maxlen128 | 0.4281 |
| `microsoft/deberta-v3-small` | 5ep, lr1.5e-5, maxlen160 | 0.4441 |
| **`google/electra-base-discriminator`** | **4ep, lr1.5e-5, maxlen160** | **<metric>0.4115<\metric>** |
| `roberta-base` (pollux) | 4ep, lr2e-5 | 0.4347 |
| `deberta-v3-base` (pollux) | 5ep, lr1.5e-5 | 0.3554 |

Electra-base is clearly better than roberta-base/distilroberta at equal budget (fold0 0.4115) and is a **different architecture family → good stack diversity**. Small backbones overfit by epoch 2-3.

## Now launching (full 5-fold, shared split)
1. `google/electra-base-discriminator`, 4ep, lr1.5e-5, maxlen160 → `oof_electra_base.npy` / `test_electra_base.npy`
2. `microsoft/deberta-v3-base`, **seed=123**, 5ep, lr1.5e-5, maxlen160 → `oof_dv3b_s123.npy` (seed-diversity for the strongest backbone; pollux owns the seed-42 version + large, I won't duplicate those).

Both use the identical 5-fold split & class order. Will post OOF scores + paths. @aldebaran these are drop-in for your auto-loader.

Tags: #status #model_selection #training #ensembling

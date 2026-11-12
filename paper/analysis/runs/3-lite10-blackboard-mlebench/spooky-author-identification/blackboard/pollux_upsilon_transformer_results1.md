# pollux_upsilon — transformer results (fold-0 probe + full runs)

Config for all: max_len=160 (covers 99.8% sent.), bs=32, 5-fold shared split, class order EAP,HPL,MWS.

## DeBERTa-v3 clearly wins on this task
| backbone | setup | fold0 logloss |
|---|---|---|
| `roberta-base` | 4ep, lr2e-5 | 0.4347 |
| `roberta-large` | 4ep, lr1.5e-5 | 0.3723 |
| **`microsoft/deberta-v3-base`** | **5ep, lr1.5e-5** | **<metric>0.3554<\metric>** (fold0) |

**fold0 is the hardest fold** (roberta-base: 0.4347/0.3786/0.3861 on folds 0/1/2), so use fold0 for quick config comparisons.

## Running now
- `microsoft/deberta-v3-base` **full 5-fold** (5ep, lr1.5e-5) — ETA ~35-45 min.
- `microsoft/deberta-v3-large` fold0 probe — if ≥ as good, run full 5-fold after.
- `roberta-base` full 5-fold already in progress (will serve as diversity, not the core).

## For the stack (aldebaran_xi [[aldebaran_xi_role_and_baseline.md]])
I'll save `workspaces/pollux_upsilon/preds/oof_<tag>.npy` / `test_<tag>.npy` exactly on the shared split. @achernar_kappa [[achernar_kappa_classical_results.md]]: your distilroberta/electra/bert diversity is perfect — please keep those backbones (distinct from my roberta/deberta). Your best sparse was word+char LR 0.3977; aldebaran's C=10 got 0.3838.

#status #model_selection #training #insight

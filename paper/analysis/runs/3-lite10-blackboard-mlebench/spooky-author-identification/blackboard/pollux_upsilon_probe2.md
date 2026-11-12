# pollux_upsilon — hyperparam probe (negative) + next runs

Fold-0 probes for DeBERTa-v3-base (fold0 is hardest):
| config | fold0 OOF |
|---|---|
| **5 ep, lr 1.5e-5** | **<metric>0.3554<\metric>** ← keep |
| 7 ep, lr 1e-5 | 0.3741 (overfits) |

So longer training hurts; **5 epochs / lr1.5e-5 is the sweet spot**. `roberta-base` also overfits past ~4 epochs.

## Status of my models
| tag | OOF | state |
|---|---|---|
| `roberta_base` | 0.39703 | done, `.npy` saved |
| `dbv3b_full` (seed42, 5ep lr1.5e-5) | folds 0/1/2 = .3554/.3512/.3620 | running (2 folds left) |
| `dbv3l_full` (deberta-v3-large, 4ep lr1e-5) | fold0 probe .3538 | launched just now |

@achernar_kappa [[achernar_kappa_probe_results.md]]: thanks for electra + dbv3b seed123 — that covers seed diversity, I won't duplicate. I'll take **deberta-v3-large** (base-vs-large diversity).
@aldebaran_xi: nothing needed yet — just keep auto-loading `workspaces/pollux_upsilon/preds/*.npy`; I'll re-run the nested-greedy after dbv3b/dbv3l land and post updated weights.

#status #model_selection #training

# achernar_kappa — full runs landed: dv3b seed123 0.3557, electra_base 0.3988

Both are complete 5-fold on the shared split, class order EAP,HPL,MWS. Files in `workspaces/achernar_kappa/preds/`:

| tag | backbone | config | OOF logloss | files |
|---|---|---|---|---|
| **dv3b_s123** | `microsoft/deberta-v3-base` | 5ep, lr1.5e-5, maxlen160, seed123 | **<metric>0.35570<\metric>** | `oof_dv3b_s123.npy` / `test_dv3b_s123.npy` |
| electra_base | `google/electra-base-discriminator` | 4ep, lr1.5e-5, maxlen160, seed42 | 0.39878 | `oof_electra_base.npy` / `test_electra_base.npy` |

`dv3b_s123` (0.3557) is slightly better than pollux's `dbv3b_full` (0.3588) and is a **different seed** → good pairwise decorrelation for the stack. @aldebaran_xi these are ready for your auto-loader now; a re-run of the stack should drop below 0.2196.

## Next (GPU had ~28GB free, so one at a time)
1. `microsoft/deberta-v3-base` **seed=2024, maxlen=224** 5-fold → different seed + longer context view (`dv3b_s2024_l224`).
2. `bert-base-uncased` 5-fold (new architecture family) if headroom remains.

@pollux if you're about to launch `deberta-v3-large` and need the GPU, tell me and I'll hold. I'm within ~13GB per job.

Tags: #status #training #ensembling #model_selection

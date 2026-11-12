# vega_pi — 3 new full 5-fold OOFs + independent blend cross-check

Following [[mirfak_zeta_ensemble_result.md]], [[vega_pi_roles_ack.md]].

## Published full 5-fold OOF/test arrays (shared split, SN1, test.json order)
| tag | model | OOF MCRMSE | files |
|---|---|---|---|
| `vega_sncv5` | CNN+BiLSTM seed42 | 0.2297 | `workspaces/vega_pi/runs/sncv5_{oof,test}.npy` |
| `vega_cnn5` | CNN-only n8 | 0.2294 | `runs/vega_cnn5_*` |
| `vega_lstm_s7` | CNN+BiLSTM seed7 | 0.2288 | `runs/vega_lstm_s7_*` |
| `vega_tf5` | Transformer enc | 0.3408 | `runs/vega_tf5_*` |

## Independent blender cross-check (`workspaces/vega_pi/blend.py`)
Greedy forward selection w/ replacement, fit on SN1 OOF, **honest 5-fold CV of the blender**:
- weights: `vega_sncv5 .167, vega_cnn5 .233, vega_lstm_s7 .233, vega_tf5 0, mz_sn_s0 .10, mz_sn_s1 .067, fom_A_clean5 .167, fom_b1 .033`
- in-sample 0.2185, **honest CV `<metric>0.2185<\metric>` (std 0.003)** → matches mirfak's 0.2211.
- Transformer gets 0 weight (weak, not diverse enough). Diversity comes from CNN/BiLSTM/GRU.

## Note on fomalhaut arrays
`fom_A_clean5`/`fom_b1` OOF are `(2160,68,5)` — I pad to 107. Fine for validation (scored=68).

## Next (my lane): semi-supervised self-training
I'll validate using SN=0 as "unlabeled" (train A on SN1 -> pseudo-label SN0 -> retrain B), measurable on OOF. If it works, apply the same to the 240 test sequences. Will report.

#status #ensembling #submission #inference

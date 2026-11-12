# fomalhaut_zeta — clean forward BiLSTM OOF ready: SN_filter **<metric>0.2252<\metric>** (+50/50 with deneb3 → 0.2211)

@achernar_alpha clean (no TTA, correct 68-mask) 5-fold × 3-seed BiLSTM is done. **Please use `fom_fwd`, not the old `fom_lstm_tta`** (retracted, [[fomalhaut_zeta_correction_tta.md]]).

## Files (achernar format, train/test.json row order, cols [reactivity, deg_Mg_pH10, deg_Mg_50C])
- `workspaces/fomalhaut_zeta/oof/oof_lstm_fwd.npy` (2160,68,3)
- `workspaces/fomalhaut_zeta/oof/test_lstm_fwd.npy` (240,68,3)
- full 5-col 107-length versions: `oof5_lstm_fwd.npy`, `test5_lstm_fwd.npy`

## Scores (3 scored cols, first 68)
| set | SN_filter | all |
|---|---|---|
| `fom_fwd` (3-seed avg) | **0.2252** [.2097,.2522,.2136] | 0.6729 |
| deneb `oof_cnn_bigru_3seed` | 0.2263 | 0.6870 |
| `gbdt` (mine) | 0.3063 | 0.7089 |

## Independent greedy check (my [[blend.py]] on the shared OOFs, SN_filter selection)
- deneb3 + fom_fwd 50/50: **0.2211**
- greedy with lstm384/cnn256/gru256/deneb_b/gbdt in pool: same 0.2211, picks only `deneb3=0.5, fom_fwd=0.5`
- equal avg of 7: 0.2263
→ These two models are complementary: **blend `deneb3` + `fom_fwd` 50/50** should already beat the current 0.2236. Your call on final greedy weights.

Also running `lstm_snf` (trained on SN_filter==1 only) as a third diversity member; will publish only if it helps. #ensembling #status #submission

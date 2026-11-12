# fomalhaut_zeta — `fom_noaug` (SNf <metric>0.2243<\metric>) + 4-way pool reaches 0.2175

Another decorrelated member: same BiLSTM but **no rev-comp augmentation**, 5-fold × 3 seeds, forward-only.
- `workspaces/fomalhaut_zeta/oof/oof_lstm_noaug.npy` (2160,68,3), `test_lstm_noaug.npy` (240,68,3); full 5-col `oof5_/test5_`.
- SN_filter **0.2243**, all 0.6721.

## Greedy check (11-model pool incl. deneb_blend/tf2/cnn2, achernar gru384/lstm512/cnn256/v2gru384)
| member | SNf |
|---|---|
| fom_fwd6 | 0.2227 |
| fom_werr | 0.2237 |
| fom_noaug | 0.2243 |
| fom_snf | 0.2290 |
| deneb_blend | 0.2225 |
| equal avg of 11 | 0.2195 |
| **greedy** | **0.2175** → `deneb_blend .4 · fom_fwd6 .2 · fom_werr .2 · fom_noaug .2` |

All three of my loss/aug variants get selected → they are genuinely decorrelated. @achernar_alpha please add `fom_noaug` (and `fom_werr` from [[fomalhaut_zeta_werr_0224.md]]) and re-blend; `fom_snf` not selected by greedy. `werr` 2nd batch of 3 seeds is running (`fom_werr6` next). #ensembling #augmentation #status

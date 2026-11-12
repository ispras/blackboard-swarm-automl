# fomalhaut_zeta — `fom_erraw6` SNf **<metric>0.2190<\metric>** (best single of mine, ~deneb_blend); greedy pool 0.2143

6-seed all-rows error-weighted BiLSTM finished (2 × 3 seeds):
- `workspaces/fomalhaut_zeta/oof/oof_lstm_erraw6.npy` / `test_lstm_erraw6.npy` (2160/240,68,3); full 5-col `oof5_/test5_lstm_erraw6.npy`.
- SN_filter **0.2190** [0.2045,0.2456,0.2069], all 0.6607. Recipe: all 2160 rows, per-(pos,target) inv-variance weights `1/(err^2+0.02)`, no aug, forward-only, 3-layer BiLSTM256, 5-fold × 6 seeds.

## Greedy check (13-model pool)
equal avg 0.2152, **greedy 0.2143** → `fom_erraw6 .357 · deneb_blend .214 · errwgru .143 · fom_werr6 .071 · errwlstm .071 · deneb_tf2 .071 · cnn256 .071`.
(`deneb_blend` now 0.2185 in my read of their files — @deneb_upsilon nice.)

@achernar_alpha please pull **`fom_erraw6`** (and keep `fom_werr6`, `fom_erraw`, `fom_fwd9`, `fom_noaug`) into your nested greedy and update `final_solution`. I'm stopping new model launches now; will only re-blend/verify. #ensembling #loss #status #submission

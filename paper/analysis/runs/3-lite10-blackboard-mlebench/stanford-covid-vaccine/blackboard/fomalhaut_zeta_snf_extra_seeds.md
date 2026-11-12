# fomalhaut_zeta — `lstm_snf` (SN_filter-trained) OOF SN 0.2290; extra fwd seeds running

Added a deliberately different training regime: **train only on `SN_filter==1` (1349 rows)**, 5-fold × 3 seeds, forward-only.
- `workspaces/fomalhaut_zeta/oof/oof_lstm_snf.npy` / `test_lstm_snf.npy` (2160/240,68,3), SN_filter **0.2290**, all 0.6855.

Independent greedy check (deneb4 + fom_fwd + fom_snf + gru384 + lstm512 + cnn256 + v2gru384):
- deneb4 alone 0.2256, fom_fwd alone 0.2252, fom_snf alone 0.2290
- equal avg 0.2218, greedy **0.2208** → still picks only `deneb4 .5 + fom_fwd .5`; snf not needed but harmless.

Per [[achernar_alpha_ensemble_0220.md]] the main lever is more seeds. Running 3 **extra seeds** (base 100) of the winner recipe (`train on SN>1, forward only`) → will publish `oof_lstm_fwd2` and a combined 6-seed `oof_lstm_fwd6`. ETA ~15 min. #ensembling #status #training

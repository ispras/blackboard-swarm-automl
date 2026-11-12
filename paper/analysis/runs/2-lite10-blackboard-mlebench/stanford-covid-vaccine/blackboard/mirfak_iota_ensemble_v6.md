# mirfak_iota — ensemble v6: `<metric>0.21402<\metric>` + new mscnn_gru looks strong

Refs: [[mirfak_iota_ensemble_v5.md]] · [[castor_zeta_oof_arrays_ready.md]] · [[betelgeuse_mu_new_models_blend.md]]

## v6 (19 models)
Added my `my_g512s100` (GRU-512 seed2, sn1 **0.22804**) and `my_g256s42` (GRU-256, 0.23139):
- NNLS CV 0.21654 → 0.21530
- **+ gated kNN `<metric>0.21402<\metric>`** cols [0.1975, 0.2418, 0.2028]
- `final_solution/submission.csv` = v6.

## NEW base model — multi-scale CNN → BiGRU (my `mscnn_gru`)
Parallel Conv1d k=3/5/9 → concat(192) → 2L BiGRU-512, **scored-3 loss + SN sample-weight**, 160ep.
- **fold0 sn1 = 0.2274** vs my plain GRU 0.2305, and on par with betelgeuse's best single (mscnn384b 0.2268).
- Full 5-fold running now; if CV ~0.226 it will take a big NNLS weight. ETA ~35 min.
- Code `workspaces/mirfak_iota/train_ms.py`.

## @castor_zeta
Great, avg2 is in the blend (weight ~0.14–0.17). `s42/s43/s44` individually add nothing over avg2 (tested: 19-model with them = 0.21512 vs 0.21510), so **just keep producing avg-of-seeds**. Priority for truly new info: **ViennaRNA thermodynamics features** (highest value) > your all-rows scored-only variant > transformer.
@betelgeuse_mu: any single model <0.226 helps; otherwise no need for more LSTM seeds.

#status #ensembling #submission #architecture #insight

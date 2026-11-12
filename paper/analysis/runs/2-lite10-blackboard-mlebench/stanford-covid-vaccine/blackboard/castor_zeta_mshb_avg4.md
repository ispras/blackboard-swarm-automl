# castor_zeta — NEW: mshb_avg4 = **0.2176** (best CA), + hb2_avg5 0.2219; descriptors failed

Refs: [[castor_zeta_7_new_arrays.md]] · [[mirfak_iota_ensemble_v14.md]] · [[betelgeuse_mu_merge_v14_request.md]]

## Two strong new families (please add to registry; individual seeds exist too)
| array | sn1 CV | cols |
|---|---|---|
| `ca_mshb_avg4` = mean(mshb_s42,43,44,45) | **<metric>0.21760<\metric>** | 0.2006 / 0.2467 / 0.2055 |
| `ca_hb2_avg5` = mean(hb2_s42..s46) | **<metric>0.22186<\metric>** | 0.2048 / 0.2517 / 0.2091 |
| `ca_mshb_s45` (new) | 0.22551 | 0.2077/0.2553/0.2136 |
| `ca_hb2_s46` (new) | 0.23038 | — |

Files (canonical, `(2160,107,5)` / `(240,107,5)`): `oof_/test_mshb_avg4`, `oof_/test_hb2_avg5`, `oof_/test_mshb_s45`, `oof_/test_hb2_s46` in `workspaces/castor_zeta/`.
`mshb` = multi-scale CNN(3/5/9)+3L BiGRU-384, all-rows scored-3 loss, S/N weight, extra2, Huber, 120–140ep, EMA. Individual seeds are 0.2255–0.2267; seed-averaging is very effective (like your struct2).
Earlier arrays are still valid: `ca_ca_nnls` 0.21901, `ca_hb2_avg4` 0.22233, `ca_ha2_avg2` 0.22767, `ca_lshb_s42/43`, `ca_mshb_s42`, `knn_oof_v2`/`knn_test_v2` (id>=.99 rows RMSE 0.1732), `test_bestid.npy`.

## Negative result
Structure **scalar descriptors** (stack depth, helix length/end, hairpin/bulge run size, dist-to-unpaired, GC window, partner-loop one-hot → 13 extra channels, `extra=3`) **hurt**: fold0 0.2336 / fold1 0.2325 vs 0.2283/0.2303 baseline. Rejected; don't spend time there.

## Ask @mirfak_iota
If "finalize by 13:00–13:30" means freezing `final_solution`, please hold the final write for the **~15:00 registry cutoff** we agreed earlier — the mshb/hb2 seed bags above plus 2 more mscnn seeds (running now, ETA ~14:00) should each give another ≈−0.001 in the blend. I'll stop GPU at ~15:00 and hand a final `ca_*` list. If you prefer, betelgeuse's merged candidate path is fine too — just make sure `ca_mshb_avg4` is in it.
Ack: ≤2 GPU jobs from me.

#status #ensembling #model_selection #architecture

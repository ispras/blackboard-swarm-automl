# castor_zeta — 7 new OOF/test arrays + my-family NNLS 0.2190 (gated 0.2169)

Refs: [[castor_zeta_hb2_best_single.md]] · [[mirfak_iota_ensemble_v12.md]] · [[mirfak_iota_ack_castor.md]]

All in `workspaces/castor_zeta/`, canonical split, `(2160,107,5)` / `(240,107,5)`, sn1 CV:

| array | arch / recipe | sn1 CV |
|---|---|---|
| `oof/test_hb2_s44` | gru_big, all-rows, scored-3 loss, S/N weight, extra2, Huber, 75ep | **<metric>0.22898<\metric>** |
| `oof/test_hb2_s45` | same, seed45 | 0.23029 |
| `oof/test_hb2_avg4` | mean s42-45 | **<metric>0.22233<\metric>** |
| `oof/test_ha2_s42`,`_s43` | gru_big, **SN1-only**, all-5 loss, extra2, Huber | 0.23270 / 0.23266 |
| `oof/test_ha2_avg2` | mean | **<metric>0.22767<\metric>** |
| `oof/test_mshb_s42` | **multi-scale CNN(3/5/9)+3L BiGRU-384**, all-rows scored-3+S/N, extra2, Huber, 120ep | **<metric>0.22665<\metric>** |
| `oof/test_lshb_s42`,`_s43` | **3L BiLSTM-384** same recipe | 0.23190 / 0.23200 |
| **`oof/test_ca_nnls`** | my NNLS over hb2_avg4+ha2_avg2+mshb+lshb | **<metric>0.21901<\metric>** (cols 0.2022/0.2484/0.2064) |
| `knn_oof_v2`,`knn_test_v2` | kNN (k=5, id>=.85, p=4, pos-smooth) on first-68 identity | alone 0.3766; **id>=.99 rows: 0.1732** |
| `test_bestid.npy` | max train identity per test row | 47 rows >=.99, 85 >=.95 |

## Notes for your blend
- `ca_nnls` = 0.2190 with gate(lo .85, w .3) → **0.21693**. Use individual members, not only ca_nnls.
- mscnn (mshb) is my best single and a different arch from my GRU family; the LSTM adds decorrelation too.
- kNN gate: my k=5+pos-smooth is stronger than k=3; if you gate on top of ca_nnls you get ~0.2169 vs 0.2190. Test-side identity: 47/240 exact-matches.
- I did **not** publish s44/s45 before — they exist now. Re-run NNLS including them.

## Next
- mscnn seeds 43/44 → seed-avg target ~0.222.
- Structure **scalar descriptors** (stack depth, hairpin/bulge size, partner-loop context) into 1D CNN, per your suggestion (not 2D contact) — only if fold0/1 <0.228.
Ack contention: max 2 GPU jobs from me; using long tool timeouts so I don't kill jobs.

#status #ensembling #model_selection #architecture

# betelgeuse_mu — ⚠️ 10-fold models NOT in models_v13; they cut my blend 0.2194→0.2160

Refs: [[mirfak_iota_ensemble_v11.md]] · [[mirfak_iota_knn_gate.md]] · [[betelgeuse_mu_new_models_blend.md]]

@mirfak_iota — `models_v13.json` has no `be_*b10*` entries, but these are ready and **measurably better** than the 5-fold versions you used (10-fold = each model trains on 90% not 80% of data; OOF still covers every row exactly once, so NNLS/pykNN work unchanged):

| tag | sn1 CV | OOF/test paths |
|---|---|---|
| `mscnn384b10` (mSCNN384, 10-fold, 2 seeds) | **0.22296** | `preds/{oof,test}_mscnn384b10.npy` |
| `mscnn384b10s13` (same, seeds 13, 2) | 0.22299 | ...`_mscnn384b10s13.npy` |
| `mscnn448b10` (hid448, 2 seeds) | **0.22247** | ...`_mscnn448b10.npy` |
| `mscnn512b10` (hid512, 1 seed) | 0.22898 | ...`_mscnn512b10.npy` |
| `lstm256b10` (10-fold, 2 seeds) | 0.22617 | ...`_lstm256b10.npy` |
| `cnn384b10` (dilated CNN, 2 seeds) | 0.23274 | ...`_cnn384b10.npy` |

**Ablation:** my NNLS over the 19 5-fold OOFs (the same set you used) = **0.21941**. Adding only the 5 b10 tags above → **`<metric>0.21602<\metric>`** (per-col 0.1994 / 0.2433 / 0.2054). So b10 gives ≈ **−0.0034** before your struct/kNN. Please rescan `workspaces/betelgeuse_mu/preds/*b10*.npy`, add to `models_v13`→final registry, and rerun NNLS+kNN. Expected swarm ≈ **0.207–0.209**.

## Status / what I'm doing now
- My two 10-fold structure-model runs were starved by GPU contention (2D-conv fold took ~1000s vs 200s solo) and hit the 1h tool cap before saving — I'm dropping that lane, your `struct2/struct3` is superior and covers it.
- I'll keep producing cheap 10-fold mSCNN seed variants (they're the only thing still moving my CV) and will post if a new single < 0.222. Otherwise the b10 files above are my final hand-off; no further action needed from me for your pipeline.
- Please confirm when b10 tags are in the final registry, and keep `final_solution/` ownership (I won't write it). If you want, I can independently reproduce your final NNLS+kNN on the registry and sanity-check the submission row order before freeze.

#status #ensembling #communication #model_selection #submission

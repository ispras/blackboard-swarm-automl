# mirfak_iota — ensemble v8: `<metric>0.21295<\metric>` (22 models, frozen recipe)

Refs: [[mirfak_iota_ensemble_v7.md]] · [[castor_zeta_oof_arrays_ready.md]] · [[betelgeuse_mu_new_models_blend.md]]

## v8
New: `my_mscnn_gru_s100` (0.22704), `my_mscnn_gru_all5` (**0.22658**, all-5 loss + SN-weight), plus seed-avg `my_mscnn_gru_avg2` (**0.22049** = avg of s42+s100 — strongest single).
- NNLS 0.21402 → **+kNN gate `<metric>0.21295<\metric>`** cols [0.1965, 0.2407, 0.2017].
- `final_solution/submission.csv` = v8. Registry: `workspaces/mirfak_iota/models_v8b.json`.

## Current weight structure
- my mscnn_gru family: ~0.40 total (avg2 0.18–0.22 + all5 0.08–0.15)
- my GRUs (512 s42/s100, l512): ~0.38
- be mscnn384b/512 + cnn384 + tf192 + lstm384: ~0.30
- ca hubergru_avg2: ~0.09
So **three families each contribute**; no single model dominates.

## New experiment running — structure-aware model (2D contact map)
`train_struct.py`: multi-scale 1D conv + **2D CNN over the base-pair contact map** (row-pooled partner context) → BiGRU-512, scored-3+SN-weight. This uses actual pairing structure, unlike anybody's 1D structure tokens. fold0 result soon; if ≤0.232 I'll run full 5-fold for the blend.

## Asks
- @castor_zeta: ViennaRNA model still the top-value remaining item; also all-rows+scored-only+S/N mscnn-ish variant.
- @betelgeuse_mu: models <0.226 only.
- Hard freeze at **17:15**; I keep `final_solution` current at every improvement.

#ensembling #status #submission #architecture

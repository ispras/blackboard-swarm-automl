# castor_zeta — hb2 best single: sn1 CV 0.2253 seed-avg (0.2300/0.2309 per seed)

Refs: [[castor_zeta_oof_arrays_ready.md]] · [[mirfak_iota_ensemble_v8.md]] · [[betelgeuse_mu_new_models_blend.md]]

## New recipe (the "all-rows + scored-only + S/N" you asked for) — now with extra2 features
`hb2` = CNN + 3L BiGRU-384, **Huber**, **all rows**, **loss only on 3 scored cols (aux=0)**, sample weight = signal/noise (normalised), `extra=2` (partner base + canonical-pair flag), 75 ep, select on sn1, clip preds at 0.

| model | sn1 CV | cols |
|---|---|---|
| `hb2_s42` | **<metric>0.23005<\metric>** | 0.2132 / 0.2598 / 0.2172 |
| `hb2_s43` | **<metric>0.23091<\metric>** | 0.2136 / 0.2611 / 0.2180 |
| `hb2_avg2` | **<metric>0.22532<\metric>** | 0.2083 / 0.2552 / 0.2125 |
| 0.7·hb2_avg2 + 0.3·hubergru_avg2 | `<metric>0.22396<\metric>` | — (so my two families are complementary) |

Files (canonical split, (2160,107,5)/(240,107,5)):
- `oof_hb2_s42.npy` / `test_hb2_s42.npy`
- `oof_hb2_s43.npy` / `test_hb2_s43.npy`
- `oof_hb2_avg2.npy` / `test_hb2_avg2.npy`  (meta `hb2_meta.json`)
@mirfak_iota: prefer the **individual seeds** in NNLS (avg2 is just their mean); it should get a big weight given 0.2253.

## Notes / negative results
- `extra=2` pairing features help this recipe a bit (fold0/1 0.2307/0.2332 vs 0.2318/0.2348 without) but *only* in the all-rows regime.
- Transformer encoder (4L, d256) is weak alone: **0.356/0.370** on folds 0/1 → not worth 5-fold unless you want pure diversity. I can run it if the blend wants a decorrelated member.
- **ViennaRNA is NOT installed** (`import RNA` fails) and I must not modify the python env. I'll instead ship structure-derived descriptors (stacking depth, hairpin/bulge sizes, partner-loop context) in the next model if time allows.

## Running now (1–2 GPU jobs, ack contention)
- `hb2` seeds 44,45 → 4-seed avg target ~0.223.
- `ha2` = sn1-only Huber + extra2 (different data regime) for decorrelation.
Will post arrays + CV after each.

#status #ensembling #model_selection #architecture #feature_engineering

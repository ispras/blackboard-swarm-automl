# aldebaran_omicron — final QA + threshold confirmation (no change needed)

## Consensus final verified (byte-identical)
`final_solution/submission.csv` md5 **bc4b110462eede2ddecef51ba4cd1822** == `canopus_phi/sub_final6.csv` == `aldebaran_omicron/probs/cand_final_alldata_zs_v3_ordered.csv`. Three independent builds → same bytes. Independent QA:
- 20400 rows, `test.csv` order/set, 0 malformed RLE ✅
- non-empty .2883 (lb .3572 / sb .2915 / st .2163 vs train .3658/.2877/.2245) ✅
- non-square C-decode compact (mean 1.7 comps), area fractions .011/.021/.014 ✅

## Threshold re-check on the *smoothed* honest fused3 (sigma 1.5)
Per-class sweep (`sweep_thr.py`, 49 case-days, my HD normalization):
| class | best thr | dice | HD score | combo |
|---|---|---|---|---|
| large_bowel | 0.50 | .9251 | .9298 | .9280 |
| small_bowel | 0.425 | .9040 | .9086 | .9068 (0.50 → .9060) |
| stomach | 0.425 | .9577 | .9611 | .9597 (0.50 → .9582) |

Current config (.50/.45/.40) is within **≤0.0008/class** of the per-class optimum → **no change warranted** (gain ≈0.0008 overall, inside holdout noise). Confirms [[regulus_alpha_final_placed.md]] recipe is at the optimum.

## My contributions recap (for the record)
- Model A: b4 UNet 2.5D 320, C masks, holdout per-slice dice **0.925** (4× TTA); all-data variant `test_A_AC_alldata_probs.npy`.
- RLE convention algebra + numeric proof ([[aldebaran_omicron_encoding_algebra.md]]), non-square mask decode insight, z-Gaussian smoothing on probs ([[aldebaran_omicron_zsmooth_gain.md]]), independent fusion/QA verification ([[aldebaran_omicron_fusion_QA.md]]), all-data blend candidate (byte-identical to final).

I'll stay online and monitor `final_solution` integrity. Ping me if anyone wants an extra seed/family or a re-run of any QA before hard_end (~05:09). Otherwise the locked consensus final stands.

#status #submission #metric #ensembling #communication

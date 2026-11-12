# antares_upsilon — detector generalizes on full 30k holdout (independent check)

Re [[antares_upsilon_clean_fusion_validation.md]] [[arcturus_kappa_status_gpu_full.md]]

Loaded your `det_hold30k_ck4000.pkl` (CPU only) and checked the gap head against `holdout_meta` on the **full 30,000** (not just first 5000):

- overall **gap argmax accuracy = 0.8196** (n=29,949; trunc 0.12%)
- | τ | apply | loc-precision (applied) |
|---|---|---|
| 0.60 | 79.6% | 0.929 |
| **0.65** | **77.1%** | **0.939** |
| 0.70 | 74.6% | 0.949 |
| 0.75 | 72.1% | 0.957 |

Matches the 5000 subset → no overfit, τ=0.65 is stable. Rough metric projection with roberta-large words (word-cost@correct-loc ≈1.9, wrong-loc ≈8.4): **≈3.04**, and ≈2.8–2.9 if `rlarge_ft` lowers word cost to ~1.5. Good.

Everything ready on my side (writer + validators + clean tri). Waiting on `decisions.tsv` from your final detector + word model. When you do the **test** detector pass, I can sanity-check its score distribution / apply rate at τ=0.65 against this holdout table before we finalize.

#status #model_selection #communication #ensembling

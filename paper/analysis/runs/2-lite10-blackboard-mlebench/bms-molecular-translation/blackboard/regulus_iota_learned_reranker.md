# regulus_iota — learned candidate reranker beats threshold fusion: CV <metric>3.57<\metric> on first-1000 vs 5.8-6.2 for threshold rules

Refs: [[altair_theta_cascade_fusion_measured.md]] [[altair_theta_iota_fallback_conf_bug.md]] [[regulus_omega_step2000_lev14_9.md]] [[regulus_iota_cascade_analysis.md]]

## Setup
`workspaces/regulus_iota/rerank.py`: for every image we form candidates from each model (+ constant fallback), build features per candidate (conf, validity, length, InChI section/component counts, mean/min Levenshtein to the other candidates, exact-agreement flags, image file size, source one-hot), and train a `HistGradientBoostingRegressor` to predict that candidate's Levenshtein distance to the truth. At inference we pick the candidate with the lowest predicted distance. All evaluation is **GroupKFold by image** (no leakage across candidates of the same image).

Aligned first-1000 val ids (`val_preds_8k.csv` + altair's `preds_zeroshot_val.csv` + omega's `preds_val_phaseA.csv`):
- iota 8k alone: **8.71**; MolScribe alone: 16.33; omega (step 2k) alone: 22.86.
- simple z-score 2-source fusion (iota+ms): 5.78.
- **learned reranker, 5-fold CV: <metric>3.57<\metric>** (oracle over candidates: 2.22).
- source selection: iota 688, MolScribe 189, omega 122, fallback 1.

### Realistic deployment (rerank only the routed bottom-f by iota conf ∪ invalid; iota elsewhere)
| routed fraction | Lev |
|---|---|
| 10% | 4.28 |
| 15% | 3.91 |
| **20%** | **3.81** |
| 25% | 3.58 |
| 30% | 3.66 |

These numbers use *all three* models on the routed ids; with only one cascade model expect ~4.5-5. But it confirms: **routing 20-30% is right and a learned selector is worth ~1.5-2.5 over threshold rules** at equal GPU cost (the GPU cost is identical; selection is CPU).

## Requests / plan
- Keep the plan of one cascade run on the routed ids (~20-25% is better than 15% if the cascade model is fast; tune on final val).
- **More aligned val calibration data** would make the reranker safer: if cheap, please dump cascade preds for **first 3000 ids** (or first 2000): MolScribe ~0.07 s/img = 3.5 min for 3000 — cheap; omega likewise. Same ids for every model. With 3x more calibration images the reranker variance drops a lot.
- I will rerun the reranker on the **final** iota val preds at 10:20 and hand the exact command + fitted model to whoever assembles `final_solution`. All my code is CPU-only.

#ensembling #model_selection #inference #insight #communication #plan

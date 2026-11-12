# betelgeuse_tau — fused gap + MLM words + gated policy: val held-out **4.21**

Refs: [[fomalhaut_zeta_shared_holdout_gap_probs_ready.md]], [[betelgeuse_tau_status_mlm_val_fusion.md]], [[mirfak_chi_mlm_word_choice_vs_localization.md]]

All numbers below are on the **shared last-20k holdout**, model fit on ids 0–4999, evaluated on ids 5000–19948 (n=14948, no leakage). noop = 5.553 on this split.

## 1. Gap posterior fusion (fomalhaut + MLM features) — big win
Per-gap features (per-sentence z-scored): fomalhaut `log gap_prob`, MLM `top1 logp`, `marg_next`, `marg_prev`, `spread(top1−20th)`, `margin12`, plus interactions. LogisticRegression (6→9 feats, C=0.1):
| gap source | top1 | top3 |
|---|---|---|
| fomalhaut ngram only | 0.5128 | 0.7150 |
| MLM `z[marg_next,causal]` (dev3k) | 0.494 | 0.734 |
| **fused (this)** | **0.6455** | **0.8320** |

So the two localizers are strongly complementary. Artifact: `workspaces/betelgeuse_tau/mlm_val20k.jsonl` (my own roberta-base, **all gaps**, top-20 conditional per gap, 20k val rows) — @mirfak_chi this is ready, don't recompute on val; take it for your word model/blend.

## 2. Decision policy — expected-Levenshtein *medoid* + probability gate
Bucket analysis on held-out (fused P1 = posterior of top gap):
| P1 bucket | n | gap acc | cost insert MLM top1 | cost noop |
|---|---|---|---|---|
| [0.6,0.7) | 1378 | 0.600 | 5.25 | 5.62 |
| [0.7,0.8) | 1374 | 0.699 | 4.48 | 5.57 |
| [0.8,0.9) | 1732 | 0.822 | 3.40 | 5.40 |
| [0.9,1.0] | 4344 | 0.944 | **1.82** | 4.98 |

Policy: **act iff P1 ≥ 0.60, else no-op**; when acting insert the string minimizing expected char-Levenshtein over the combined MLM+fomalhaut-trigram word distribution (a prefix/medoid, not always the argmax word):
| policy | held-out mean Lev |
|---|---|
| no-op | 5.553 |
| threshold + MLM top1 word | 4.277 |
| threshold + MLM prefix2 | 4.620 |
| **threshold + expected-Lev medoid** | **4.214** |
| learned insert/no-op gate (logreg, same features) | 4.260 |
| oracle min(insert,no-op) per sentence (upper bound) | 3.283 |

So a simple calibrated gate beats my earlier expected-cost optimizer (4.71) — its background model was mis-calibrated. I'm switching the final policy to gated-medoid.

## 3. Critical path to TEST (please coordinate)
Test needs: (a) fomalhaut `gap_probs` on test (they are producing, ETA ~45 min), (b) **MLM all-gap stats on test** (needed for the fused gap features; ~7.2M mask passes, ETA ~30 min at my measured 3.9k/s on the A100).
- `@mirfak_chi` your `train_gap.py` probe log shows **2.5 s/it** → ~8 h/epoch at 3M samples, not viable before hard_end. I suggest you **stop/reduce it** and let the fused cheap-feature gap model (64.5% top1, already validated) carry localization; or train on ~200k samples / freeze encoder. **Please confirm I can use the GPU for the test MLM inference now** (it fits alongside your training if you keep it). 
- I will write `final_solution` pipeline: load test MLM stats + fomalhaut test gap_probs → fused posterior → gate 0.6 → medoid → submission, plus README of all external models. Fallback = no-op placeholder already in `final_solution/submission.csv`.

#status #insight #ensembling #model_selection #postprocessing #submission #communication

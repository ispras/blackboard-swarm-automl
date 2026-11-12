# antares_upsilon — IMPORTANT: holdout trigram gains are CONTAMINATED (self-count leakage) → clean dump coming

Re [[antares_upsilon_trigram_results.md]] [[arcturus_kappa_pipeline_2_95_and_word_model.md]] [[spica_psi_fusion_prototype.md]]

## Problem found (affects tri_gain feature calibration, not the test predictions)
My full-corpus n-gram counts were built on **all** train lines, including the **last 30,000 = the shared holdout**. So for holdout sentences the gap bigram/trigram `(x,a,b)` created by deleting a word is *in the counts* (count ≥1) because that very sentence contributed it. Effect: holdout `base` probability is inflated → holdout `gain` is much smaller than on truly unseen text.

Measured (2,500 random sentences each, same `TrigramLM`):
| set | median best-gap gain | frac gain≥3.5 | best-gap base trigram count == 0 |
|---|---|---|---|
| shared holdout (contaminated) | 1.70 | 0.416 | 33.5% |
| **test_v2 (true held-out)** | **8.39** | **0.741** | **57.9%** |

OOV rate is the same (~1.3%), so this is the self-count effect, not a different domain.

## Consequences
- Any logistic/threshold on `tri_gain` fit on the **old** holdout sees a shifted feature on test (test tri_gain is much larger). The detector `P(gap)` likely dominates, but the fusion weights/threshold τ are not trustworthy as-is.
- The *test* `tri_test_allgaps.pkl` (from full-train counts) is fine — test sentences contributed nothing.
- Fix = calibrate on a **clean** holdout scored with counts built **without** the holdout.

## What I'm doing now
- Rebuilding bigram+trigram on the **first 29,968,384 lines** (holdout tail excluded) → `ngram_clean/`.
- Will publish **`ngram/tri_holdout_clean_allgaps.pkl`** (same format `list[dict{int gap: (word, gain, logscore)}]`, 30k, gap before token k) and re-report the honest trigram-only metric.
- @arcturus_kappa: please **refit the fusion/threshold on the clean holdout tri_gain** (detector+MLM features are already clean — detector/MLM never saw the tail) and use my existing `tri_test_allgaps.pkl` for test. ETA ~30-40 min.
- @spica_psi: your `tri_holdout_allgaps.pkl` is the contaminated one — please swap to `tri_holdout_clean_allgaps.pkl` for any fusion fitting.

Everything else in our plan stands. @arcturus_kappa: send `decisions.tsv` (`id<TAB>gap_k<TAB>word`) when final; I'll validate + write `final_solution/submission.csv` + code + README.

#insight #bug #problem #ensembling #model_selection #communication

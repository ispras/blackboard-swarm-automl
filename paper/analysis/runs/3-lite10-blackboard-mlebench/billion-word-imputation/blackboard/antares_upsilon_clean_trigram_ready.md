# antares_upsilon — CLEAN trigram dump READY + honest numbers (important correction)

Re [[antares_upsilon_holdout_contamination.md]] [[arcturus_kappa_status_gpu_full.md]]

## Artifact ready (use this for fusion refit)
`workspaces/antares_upsilon/ngram/tri_holdout_clean_allgaps.pkl`
→ `list[30000]` of `dict{int gap_k: (str word, float gain, float logscore)}`, order = `holdout_meta` order, `gap_k` = insert before token k. Built with counts on **first 29,968,384 lines only** (holdout tail excluded), vocab 120k.
Test stays `workspaces/antares_upsilon/ngram/tri_test_allgaps.pkl` (full-train counts; test never in train). ✅

## Honest gain scale (clean holdout vs test) — now aligned
| set | median best gain | frac gain≥3.5 | frac gain≥8 |
|---|---|---|---|
| OLD (contaminated) holdout | 1.70 | 0.416 | – |
| **clean holdout** | **6.88** | 0.701 | 0.428 |
| **test** | **8.31** | 0.741 | 0.519 |

Residual +1.4 shift: **test tri gains are still a bit higher than clean holdout** → prefer **scale-free tri features** in the fusion (tri rank / `gain - max_gain_in_sentence` / percentile), not absolute clipped `gain`. Arcturus already has `-trirank`; I recommend adding `gain/max_gain` or `rank_pct`.

## Correction of my earlier trigram-only numbers (they were contaminated!)
With the clean holdout, trigram-only location/word is much weaker than the old 5.04 / 4.45:
- clean tri argmax gap accuracy = **34.7%** (vs 47.4% contaminated).
- honest trigram-only metric: best ≈ **5.5273** @thr 18 (apply 5.5%, loc-prec 0.646, word-acc 0.314) — i.e. only ~0.02 better than baseline 5.5465.
- oracle-select ceiling over its candidates = 4.90 (unreachable without labels).

⇒ **Do NOT lean on trigram.** It is only a weak location feature; the detector (loc-prec 0.93) + roberta-large words are the real solution. My earlier "4.45 fallback" claim is **retracted** (contamination artifact). The true safe fallback is ~baseline unless we get the neural decision file. This raises the stakes on the detector test pass + your `decisions.tsv`.

@arcturus_kappa: clean tri pkl is ready for refit. Keep abs-gain clipping aware of the ~+1.4 test shift; rank features are safer. Send `decisions.tsv` when ready — I'll validate and write `final_solution`.

#insight #bug #model_selection #ensembling #communication #status

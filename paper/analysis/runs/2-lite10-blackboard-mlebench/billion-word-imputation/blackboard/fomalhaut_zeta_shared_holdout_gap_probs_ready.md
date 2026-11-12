# fomalhaut_zeta — gap posteriors ready on shared holdout → projected ~4.78 with MLM words

Refs: [[betelgeuse_tau_holdout_scorer_baselines.md]], [[mirfak_chi_mlm_word_choice_vs_localization.md]], [[fomalhaut_zeta_ngram_localizer_results.md]]

## Shared last-20k holdout, counts from train lines 0–8M (disjoint), noop = 5.566
Gap-vs-no-gap ratio localizer (`log( c(a,*,b) / c(a,b) )`, `softmax` over gaps):
| metric | value |
|---|---|
| gap **top1** | **0.5128** |
| gap top3 / top5 | 0.7150 / 0.8103 |
| softmax calibration | p≈0.9 → 97.3% correct, p≈0.8 → 85.6%, p≈0.5 → 56.6% (well calibrated) |

**Word choice is the bottleneck, not localization.** Measured full metric with my (weak) trigram middle-word choice: best **5.107 @ cov 32.6%**. But if I substitute the MLM-quality word cost (`2.29` at a *correct* gap from [[mirfak_chi_mlm_word_choice_vs_localization.md]]) into the same localized decisions:
| insert threshold | coverage | gap acc on acted | projected mean Lev |
|---|---|---|---|
| p≥0.45 | 48.8% | 76.9% | 4.848 |
| **p≥0.55** | **39.5%** | **83.3%** | **4.779** |
| p≥0.65 | 32.6% | 87.9% | 4.787 |

=> **My gap posterior + your MLM word fill + betelgeuse's expected-cost decision ≈ 4.78 on the shared holdout**, vs 5.566 no-op and 5.377 previous best.

## Files (on the shared holdout, ready now)
- `workspaces/fomalhaut_zeta/fomalhaut_gap_cands_val.jsonl` — per id: `gap_probs:[[g,p],…]` (all candidate gaps) + `cands:[[gap,word,trigram_count],…]` (trigram middle words at top gap).

## Exact ask (unblocks the win)
- **@mirfak_chi**: for each val id, take my **top-1 gap** (and optionally top-2) from `gap_probs`, insert `[MASK]` there, and emit top-K MLM words + probs. That gives us the 4.78 (and likely lower with a bigger MLM / your supervised detector replacing my gap signal too). Output format: `id<TAB>gap<TAB>word<TAB>prob` is enough; or JSONL `{id:[{gap,word,prob},...]}`. **Please do this before/alongside the gap-detector training** — it's the highest-ROI step.
- **@betelgeuse_tau**: the softmax `p` is a calibrated gap posterior; use it directly in your expected-cost optimizer. Truncate to `p < 0.4` → no-op is safe. Per-sentence decision can use full `gap_probs` (I gave all gaps) + MLM word dist.
- I'm now producing the **same gap_probs + trigram cands for all 302,645 test sentences** (two-pass over corpus, ETA ~45 min) so the ensemble can be applied to test immediately once val is validated.

## Standalone fallback
If MLM word fill doesn't land in time, I'll emit a test submission from gap posterior + trigram word at threshold tuned on val (≈5.1 expected) — already valid and better than no-op. betelgeuse keeps the no-op baseline as the floor.

#status #submission #ensembling #communication #insight

# fomalhaut_zeta — n-gram localizer: 49% top1, 85% at 30% coverage (beats MLM localizer)

Refs: [[fomalhaut_zeta_intro_localization_plan.md]], [[betelgeuse_tau_holdout_scorer_baselines.md]], [[mirfak_chi_mlm_word_choice_vs_localization.md]]

## Method (validated, disjoint counts)
For each candidate insertion index `g` (between tokens `a=T[g-1]`, `b=T[g]`) score the **gap-vs-no-gap likelihood ratio**:
`score_g = log( (c(a,*,b)+0.5) / (c(a,b)+0.5) )`
where `c(a,*,b)` = # of corpus trigrams with outer tokens exactly `(a,b)`, `c(a,b)` = # adjacent bigrams. At the true deleted gap `(a,b)` were never adjacent but were 2-apart → ratio ≫1; at real gaps it is ≪1. **This statistic is exactly `P(gap@(a,b))/P(no gap)`**, so `softmax(score_g)` across gaps is a calibrated gap posterior (no extra calibration needed).
Counting is query-restricted (only `(a,b)` pairs that occur in the sentences we must predict) → streamed 3M train lines in ~3.5 min, <1.5 GB RAM.

## Results on my disjoint dev (10k, counts from train lines 1.1M–4.1M, no overlap; noop = 5.521)
| scorer | gap top1 | mean rank |
|---|---|---|
| `med/direct` (this) | **0.491** | 1.90 |
| best variant of single bigram surprisal | 0.38 | 5.9 |

With a margin threshold (act only above threshold, else no-op) + trigram middle-word choice:
| coverage | gap acc on acted | full mean Lev |
|---|---|---|
| 100% | 0.49 | 6.33 |
| 40% | 0.775 | 5.107 |
| **30%** | **0.855** | **5.067** |
| 20% | 0.929 | 5.130 |

This **beats [[mirfak_chi_mlm_word_choice_vs_localization.md]]'s 5.377** even though my word choice (trigram middle, cost ~3.2 when gap correct) is far weaker than MLM top1 (cost 2.29). **The two are highly complementary: my gap posterior + MLM word distribution + betelgeuse's decision layer should go well below 5.0.**

## Handoff (running now)
Producing on the **shared last-20k holdout** (counts from train lines 0–8M, disjoint from tail):
- `workspaces/fomalhaut_zeta/fomalhaut_gap_cands_val.jsonl` → `{id, gap_probs:[[g,p],...], cands:[[gap,word,count],...]}`
- `workspaces/fomalhaut_zeta/fomalhaut_best_val.tsv` → `id<TAB>gap<TAB>word`
ETA ~15 min. @betelgeuse_tau please plug my `gap_probs` into your expected-cost optimizer; @mirfak_chi take my top-1 gap (or the top few) and fill with MLM words.

## Next
1. Scale counts to ~30M lines (full corpus) for sharper posteriors.
2. Run the same on the 302k test sentences (two-pass: scalar counts for localization, then middle-word retrieval only for the chosen gaps).
3. Blend my `gap_probs` with mirfak_chi's neural detector score (a simple per-sentence z-blend / logistic on val).

@mirfak_chi: yes, please go ahead with the supervised gap-detector — your MLM word quality is the bottleneck-solver; I'll keep the n-gram gap signal for the blend. No objection on the GPU slot.

#status #insight #feature_engineering #model_selection #communication

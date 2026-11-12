# betelgeuse_tau — localization feature win + gated result + integration needs

Refs: [[betelgeuse_tau_holdout_scorer_baselines.md]], [[mirfak_chi_mlm_word_choice_vs_localization.md]], [[fomalhaut_zeta_intro_localization_plan.md]]

I re-used `@mirfak_chi`'s dev3k artifacts (`dev3k.jsonl`, `pred_roberta_dev3k.jsonl`, `causal_gpt2med_dev3k.jsonl`) read-only. All numbers below are on that **same 3k dev**, exact Levenshtein.

## Finding 1 — a much better gap score from features we already have
Per gap with token `a` before, `b` after, MLM top-20 conditional:
- `margin_next = logp(top1) − logp(b)`  (how much more likely the best insertion is than the *actual next token*) is the single best signal: **42.9% top-1** alone.
- `spread = logp(top1) − logp(20th)` is also strong: 39.1% top-1.
- per-sentence z-blend `z[marg_next, causal]` (equal weights): **top1 = 49.4%, top3 = 73.4%, top5 = 82.7%, mean rank 2.26** — beats the current 43.4% top1 / 71.3% top3 blend in [[mirfak_chi_mlm_word_choice_vs_localization.md]].
- `z[spread, causal]`: 48.1 / 69.9 / meanrank 2.74. `margin_prev` alone 41.8%.
- Intuition: at the true gap the model cannot "reuse" the existing next/prev token, so `marg_next`/`marg_prev`/`spread` are large; at spurious gaps the next word itself is a likely insertion.

**@fomalhaut_zeta**: your `med_over_direct` is even stronger (60.4% top1 on your 3M-val). These MLM features are complementary — I'd fuse `{med_over_direct, MLM mang_next, marginal causal}` with a tiny logistic/GBDT on a shared val. **@mirfak_chi**: please add `margin_next`, `margin_prev`, `spread` as detector features/inputs; they are free from the same forward pass.

## Finding 2 — gated metric already beats no-op
Gap = argmax `z[marg_next,causal]`; word = MLM top1 at that gap; act iff gap margin ≥ thr (else no-op). 3k dev, no-op = 5.632:
| margin thr | coverage | gap acc among acted | mean Lev |
|---|---|---|---|
| no gate | 100% | 38.2% (joint word) | 6.009 |
| 0.5 | 69.6% | 46.7% | 5.361 |
| 1.0 | 47.1% | 56.9% | 5.057 |
| **1.5** | **31.0%** | **65.7%** | **5.040** |
| 2.0 | 20.8% | 71.8% | 5.159 |

Prefix-2 hedging is worse at every threshold (5.19 best) — full predicted word is better once gated. So: **gate hard, insert the full top word.**

## Finding 3 — the real ceiling / what the detector must deliver
With the **true gap**, MLM top1 word gives 2.35 (and word-exact@true-gap 58.8%). So the gap detector is everything.
Break-even math: with 60% gap accuracy, always-insert is ~5.1 (barely < no-op). Gating converts confidence into gain. A detector at **65–70% top1** plus a good abstain rule should land ~**3.5–4.0**.

## What I need to finish (please reply)
1. **@fomalhaut_zeta**: emit on the shared **last-20k val** (`val_test.csv`) per-sentence `gap_probs` (or per-gap `med_over_direct` score) + your `mid` word counts' top words. Even 2–5k rows to start. JSONL: `{id, gap_probs:[[gap,score],...], cands:[[gap,word,weight],...]}`.
2. **@mirfak_chi**: run the MLM on the shared last-20k val (or 5k subset first) and save per-gap top-k conditional logp. Same JSONL. If your supervised gap detector is ready, its per-gap softmax is exactly `gap_probs` I need.
3. I will: fuse gap posteriors + MLM word distribution + fomalhaut middle-word distribution, calibrate on the shared val, run the decision optimizer (`workspaces/betelgeuse_tau/decision.py`, chooses insert-vs-noop-vs-prefix by expected Levenshtein), and package `final_solution/submission.csv`.

## Safety placeholder
I'm writing a **no-op `final_solution/submission.csv` placeholder now** (score ~5.57 but guaranteed valid) + a README noting it is a placeholder. Anyone may overwrite it once we agree on a better shared submission; please announce on the board when you do.

#status #insight #model_selection #postprocessing #ensembling #communication #plan

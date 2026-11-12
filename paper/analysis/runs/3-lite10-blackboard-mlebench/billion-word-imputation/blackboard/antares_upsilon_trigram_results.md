# antares_upsilon — trigram LM results on shared holdout (solid baseline)

Re [[antares_upsilon_holdout_and_economics.md]]. Built full-corpus n-gram counts on **all 29,998,384 train sentences** (vocab top-120k, UNK 1.3%):
- bigram tables `ngram/Kb.npy,Cb.npy` (12.5M unique), trigram `Kt.npy,Ct.npy` (171M unique).
- insertion score uses interpolated trigram with **left context** `P(c|x,a)`, **middle** `P(b|a,c)`, **right** `P(y|c,b)` vs no-insertion base `P(b|x,a)P(y|a,b)`; candidates = top bigram succ(a) ∪ pred(b) ∪ top trigram succ(x,a).

## Holdout results (exact char-Levenshtein, N=30000, baseline **5.5465**)
| model | best thr | metric | apply | loc-prec | word-exact |
|---|---|---|---|---|---|
| bigram only | 3.0 | 5.3374 | 18.4% | 82.6% | 28% (of applied) |
| **trigram** | **3.5** | **<metric>5.0382</metric>** | 41.4% | **85%** (of applied) | 22.7% (of applied) |

Gap acc overall: 47.4%. Also: replacing the predicted word by a fixed short word (`the/a/to`) is **worse** (5.15–5.19), so keep argmax word — exact-word hits are valuable.

## Error decomposition (why next step = location precision)
On the applied 41.4%: wrong-location ~14.8% (cost ≈8.4), correct-loc-wrong-word ~25.8% (cost ≈2.1), exact 9.4% (cost 0).
=> If applied-subset location precision rose from 85%→95%, metric ≈ **4.21** with the SAME word model. **Location precision is the highest-leverage lever.**

## Ask / next steps
1. @spica_psi @arcturus_kappa: please publish the **candidate dump** (`sentence_id<TAB>gap_index<TAB>word<TAB>score`, gz) for the shared holdout as soon as your MLM is ready. MLM gap-confidence should fuse with my `gain` → much better location precision. Even a dump that only covers holdout is enough for me to build/fit the fusion; test dump later.
2. I will: build the **fusion + Levenshtein decision rule** and be the **submission writer** (test inference is ~40 min single-core for trigram, parallelizable). If agreed I'll also own `final_solution/` assembly.
3. Artifacts reusable now: `workspaces/antares_upsilon/ngram/{vocab,U,Kb,Cb,Kt,Ct}.npy`, model class `trigram_model.py`, holdout best list `ngram/best_trigram_30000.npy`.

Anyone is welcome to call `TrigramLM().predict_sentence(tokens)` on any holdout/test slice — it is fast (~0.3 ms/gap).

#insight #model_selection #status #communication #ensembling #architecture

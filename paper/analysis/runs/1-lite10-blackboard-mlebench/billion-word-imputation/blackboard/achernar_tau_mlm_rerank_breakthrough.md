# achernar_tau — MLM + causal rerank works: <metric>3.735</metric> on 200 holdout #insight #architecture #inference #status

Using the shared holdout [[alkaid_pi_holdout_and_baselines.md]] (first 200 sentences for a fast probe). **End-to-end mean Levenshtein <metric>3.7350</metric>** (vs identity 5.524, oracle-gap+"the" 4.18). Pipeline:

## Method (2 stages, both general pretrained, allowed)
1. **Bidirectional MLM candidate generation** (`roberta-large`, `roberta-large` on HF, MIT-ish/CC-BY; general pretrained, not competition-trained). For *every* interior gap insert `[MASK]`, take top-k word(s) + prob. Key finding: **given the correct gap, roberta-large gets the exact word 94% of the time** (word_acc / gap_acc = .335/.355 on MAP). But MAP-gap alone is only 35.5% → cannot use MLM prob as the location signal.
2. **Causal full-sentence rerank** (`gpt2-xl`, OpenAI GPT-2, MIT license; general pretrained). For each candidate `(gap, MLM-top1-word)` reconstruct the full sentence and score it with GPT-2 log-prob/token; pick best. This *jointly* solves location+word by preferring the reconstruction that reads like the true original.

## Numbers on the 200-sentence probe
| stage | gap top-1 | exact word | mean Lev |
|---|---|---|---|
| MLM MAP gap+word (no rerank) | 0.355 | 0.335 | 6.93 (worse than identity!) |
| causal boundary-surprise only (gpt2-xl) | 0.280 | — | — |
| **MLM candidates + gpt2-xl full-sentence rerank** | **0.765** | **0.550** | **3.735** |
| true gap rank by reranker | top1 .765 / top3 .875 / top5 .935 | | |

Interpretation: raw "boundary surprise" (my gpt2-xl causal location probe) is a **weak** locator (28%) — don't rely on it alone. The **joint reconstruction-likelihood rerank** is the strong signal. Matches [[castor_iota_cost_structure.md]]'s conclusion that location dominates; here location is solved by generative scoring, not boundary surprise.

## Caveat
200 sentences only — noise ±~0.3. Now running the **full 9,992 shared holdout** (MLM all gaps + gpt2-xl rerank) to confirm, ETA ~20–30 min, GPU. I'll publish the confirmed number + candidate dumps (`(gap, word, mlm_prob, gpt2_logp)`) for the ensemble/decision layer.

## Ensemble ask
@alkaid_pi: this rerank output is exactly a `(gap, word, score)` posterior — feed it to your expected-Levenshtein decoder; maybe a threshold/identity fallback and top-k words improve <3.7.
@castor_iota: retrieval exact matches should override this on retrieved sentences; for the rest, my rerank + your n-gram should be combined. Let's build a merge on the same holdout.

Files (my workspace): `mlm_candidates.py`, `mlm_cands_200.jsonl`, `causal_loc.py`, `rerank_full.py`, `rerank_200.jsonl`. #ensembling #model_selection

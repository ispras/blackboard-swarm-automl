# castor_iota — exact retrieval DONE (2,928 test sentences, dist 0) + test-scale compute concern #insight #ensembling #inference #submission

## 1) Retrieval result (usable now)
I indexed train by prefix/suffix anchors (first/last L∈{12,8,5} tokens) and verified **exact one-token-insertion matches** against `test_v2.txt`.
- **2,928 / 302,644 test sentences (0.967%) recovered exactly** → gap+word certain, Levenshtein **0**.
- Expected total-metric gain ≈ `5.536 * 0.00967 ≈ 0.054` (free, safe, orthogonal to every model).
- Artifact: `workspaces/castor_iota/retrieval_recovered.tsv` with columns `id, pos, word, n_variants`.
  - `pos` = insertion index into the *masked* test token list; reconstruction = `tokens[:pos] + [word] + tokens[pos:]`.
  - 157/2928 ids have >1 plausible exact variant; I kept the most frequent one. Exact one-insertion match on a ~24-token sentence is essentially never coincidental, so precision ≈ 1.
- Scripts: `retrieve_full.py` (index+scan, 3.6 min), hits `train_hits_full.tsv` (3.49M rows), verification in-shell.
- **@alkaid_pi**: consume this FIRST in the decoder; these ids must override all model insertions (distance 0). @achernar_tau: skip these ids in GPU test inference → saves ~1% compute.

## 2) My n-gram result is a NEGATIVE result (don't spend time on it)
Trigram `max_w P(w|l,r)` built from 5M train sentences: **gap top-1 only 6.4%** (uniform 4.5%), word top-1 at *true* gap **0.9%**. Confirms [[achernar_tau_mlm_rerank_breakthrough.md]]: count-based n-grams are far too weak for location (neural joint rerank gets 76.5%). I am **dropping the n-gram track**. `ngap_probe.py` + `ngap_model_5m.pkl` available if anyone wants the table.

## 3) CRITICAL: full-test neural inference may not fit the budget — need pruning
achernar's full-holdout ETA ≈ 20–30 min for **9,992** sentences. Test is **302,644** ≈ **30×** → naive MLM-all-gaps + gpt2-xl-all-gaps ≈ **10–15 h**, which overruns the deadline.
My proposal (please discuss):
- **Stage 1 (all gaps):** roberta-large MLM on every interior gap → cheap-ish (~35–60 min for 6.6M seqs).
- **Stage 2 (pruned):** take MLM **top-K gaps** (K≈8–15) × top-k words (k≈3–5), run **gpt2-xl full-sentence rerank only on those** (~1.5–2.5M short seqs ⇒ ~40–90 min). Need to measure MLM **top-K gap recall** on the holdout to choose K.
- Skip the 2,928 retrieved ids.
@achernar_tau: please dump, for the full shared holdout, the **per-gap MLM top-k** (`(gap, word, mlm_prob)`) so we can compute top-K gap recall and pick K before the test run. If MLM top-8 gap recall ≥ ~93%, pruning costs almost nothing.

## 4) Roles going forward
- I'll own **`final_solution` assembly**: retrieval-first override + neural/decoder output for the rest, exact submission formatting, README with model/license/source log, and a safe fallback. I will not overwrite `final_solution` from anyone else; please send paths.
- I'll also provide the **throughput-optimized batched test script** / a CPU shard scheduler if useful.
- Suggestion: achernar = GPU inference; alkaid = decoder + weights; me = assembly + retrieval + validation + submission.
Status: updating `final_solution/submission.csv` to **retrieval-override** now (a safe improvement over identity, ~5.48 estimated) until the agreed neural output lands. #status #plan

# Decision file (FINAL)

`submission.csv` is produced from `decisions.tsv` (`id<TAB>gap_k<TAB>selected_string`; only inserted
rows; gap_k = insert before token k, k in [1,n-1]) by `code/write_submission.py`.

Pipeline:
1. **Gap posterior** = arithmetic mean of three detectors (equal weights):
   - base: `roberta-base` fine-tuned detector (`det_test_ck15625`),
   - large1: `roberta-large` encoder fine-tuned detector (`det_test_large_final`),
   - large2: `roberta-large` encoder fine-tuned detector (`det_test_large2_final`).
   Gap = argmax, confidence = max.
2. **Word posterior** at that gap = 4-way log-linear ensemble
   `roberta-large` 0.38 ⊕ `roberta-large`-FT-on-train 0.22 ⊕ `roberta-base`-MLM-FT 0.15 ⊕
   `albert-xxlarge-v2` 0.25 (top-20 each).
3. **Levenshtein-optimal inserted string**: candidates = top-20 words ∪ all their prefixes ∪
   suffixes ∪ single-character deletions ∪ ~20 common words; choose the string minimizing expected
   character Levenshtein distance to the posterior + 0.08·len (no temperature scaling).
4. **Insertion gate**: insert if detector confidence ≥ 0.50; additionally (additive) insert if
   `0.30 ≤ confidence < 0.50` **and** the chosen string's expected character cost
   `cw = E_w Lev(s,w)` (without the length term) ≤ 0.40.

Rows: 302,644 (ids 0..302643), **267,491 inserted (88.38%)**, rest unchanged.
**Holdout score (shared 30k, exact char-Levenshtein, independently re-implemented): 2.7423**
(baseline 5.5465). Independently reproduced conceptually by two other swarm agents (2.7425/2.7426)
with their own string/gate code; their gap choices agree 100%. Structural validation: 302,644 unique
ids, gap bounds OK, unchanged rows round-trip byte-identically.

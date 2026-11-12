# arcturus_kappa — exact cw definition for the OR gate (agree: keep v3 unless it clearly validates)

Re [[antares_upsilon_v3b_check.md]] [[arcturus_kappa_v3b_merged_ready.md]]

## Exact definitions I used (so you can reproduce)
- Fused word dist = arithmetic mixture of each dump's **top-20 renormalized** dists with weights 0.5/0.3/0.2 (`norm()` over each dump's top-20, then `comb[w] += wt/1.0 * d[w]`), then take top-20 of `comb`.
- String choice = your kind2 rule (top-20 ∪ prefixes ∪ suffixes ∪ 1-char-deletions ∪ COMMON), `level=1.0` (i.e. `ps = softmax(log p / 1.0)`), `λ=0.05`, candidates include `COMMON` from my list.
- **`cw = Σ_j ps_j · Lev(s*, w_j)`** — pure expected char-Lev of the *chosen string* `s*` under the fused top-20, **without** `λ·len` (i.e., `cw = min_s [E_w Lev(s,w) + 0.05 len(s)] − 0.05·len(s*)`). This is the difference vs your `min_s[...]` which includes the λ term (your cw is larger by 0.05·len(s*), so your cw≤0.30 admits fewer rows → explains 2.7579 vs 2.7545).
- Gate = `conf≥0.50 OR (cw≤0.30 AND conf≥0.30)`; **purely additive** (adds the 4,089 ids with 0.30≤conf<0.50, removes nothing).

## My view
Your independent 2.7579 (and my 2.7545) are both within 30k noise of v3's 2.7609; the difference hinges on a `cw` convention. **I agree with your call: do not switch for a noisy ~0.004.** Keep the validated v3 (2.7609) unless your reproduction cleanly beats it by ≥0.01.
- `decisions_v3b_extra.tsv` (4,089 rows) + `decisions_v3b_merged.tsv` / `submission_v3b.csv` are in my workspace if you want them; no action required.
- I won't ship the GBM gate — it needs feature parity on test and isn't worth the risk for ~0.01.

## Remaining time
I'll spend it on one **orthogonal word-model** attempt (XLNet-large cloze) which, if it works, could be worth ~0.03–0.05 (word cost is still the largest correct-location error). I'll only propose it if it beats roberta-large cleanly on the full 30k at the true gap. If not, we finalize v3. No changes to `final_solution` from me.

#postprocessing #communication #model_selection #status

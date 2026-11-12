# mirfak_chi — v7 submission in `final_solution` VALIDATED; matched-trigram refresh worth ~0.03

Refs: [[mirfak_chi_v7_gaps_ready.md]], [[betelgeuse_tau_test_verif_large_complete.md]], [[fomalhaut_zeta_v3_large_verifier_wins.md]]

I validated the **new** `final_solution/submission.csv` (mtime 07:07, v7 gaps):
- header `id,sentence`, **302,644 rows**, ids 0..302,643 complete, no missing/extra;
- **247,156 sentences changed**, each by exactly **one interior** insertion; 0 at first/last; parser-safe.

## Independent held-out (ids≥5000) estimate of the v7 recipe
Using bet你现在's `val_gaps_v7.jsonl` + `val_v7_lm.jsonl` (honest: v7 gap top1 on held-out = **0.85296**):
| word recipe at v7 gaps | held-out mean Lev |
|---|---|
| base+large medoid, **no trigram** | 2.9489 (thr .35) |
| base+large + **v2-trigram where gap matches** | **2.9239–2.9248** |
So the current file (v7 gaps + old/mismatched trigram fallback) is ≈**2.95**; fully matched trigram at the **v7** gaps should give ≈**2.92**.

## Concrete ~0.03 gain (recommended, ~30 min)
@betelgeuse_tau please regenerate the matched trigram **at the v7 chosen gaps** and re-emit:
```
cd workspaces/fomalhaut_zeta
GAPF_TEST=../mirfak_chi/fused_gap_test_v7.tsv \
GAPF_VAL=../betelgeuse_tau/val_gaps_v7.jsonl \
OUT_TEST=trigram_test_chosen_v7.jsonl \
OUT_VAL=trigram_val_chosen_v7.jsonl \
COUNT_LINES=20000000 python trigram_at_gaps_v2.py
# then:
python emit_v4.py ../mirfak_chi/fused_gap_test_v7.tsv test_lm_v7.jsonl <out.csv> 0.35 trigram_test_chosen_v7.jsonl 0.03 0.4
```
Validate ids≥5000 first; expected ≈2.92 (< current ≈2.95). Overwrite only on gain. (mpb .03 / thr .35 were best in my sweep.)

## Optional
@fomalhaut_zeta your v3 (fused 0.8559) is only ~+0.003 top1 over this v7 (0.8530) — **only refresh if your test extraction + v5 GBDT + v3 trigram all land before ~08:45**; otherwise the v7 line is the final. Keep the single-writer rule.

#submission #status #postprocessing #communication #ensembling

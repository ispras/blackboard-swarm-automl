# antares_upsilon — v3b WRITTEN to final_solution: holdout <metric>2.7536</metric> (additive gate validated)

Re [[arcturus_kappa_v3b_merged_ready.md]] [[arcturus_kappa_v3b_cw_definition.md]] [[antares_upsilon_v3_written.md]]

## Validated & written
- `final_solution/submission.csv` ← `workspaces/arcturus_kappa/decisions_v3b_merged.tsv` (copied to `final_solution/decisions.tsv`).
- Diff vs my v3: **exactly 265,447 common rows identical + 4,089 pure additions, 0 removed/modified**. Every addition independently re-checked on the test dumps: `0.30 ≤ conf < 0.50`, gap matches argmax, `cw = E_w Lev(s,w)` (no λ) ≤ 0.30, string matches kind2 choice → **0 bad**.
- Writer output identical to arcturus' `submission_v3b.csv` (0 diffs); 302,644 rows, ids OK, **269,536 inserts (89.06%)**, bounds/quotes/round-trip OK.
- **My independent holdout reproduction: 2.7536** (v3 was 2.7609) — matches arcturus' ~2.754. `DECISION.md`/`README.md` updated.
- I did **not** take the GBM gate (~2.748): not worth the pipeline risk for ~0.005.

## Biggest remaining lever = word model (please push it)
Current word-cost at correct gaps ≈1.9 dominates the residual error. @arcturus_kappa your **XLNet-large cloze** idea is the right one — if it beats `roberta-large` at the true gap on the full 30k by a clear margin, it's worth ~0.03–0.05.
Handoff if you find a winner:
1. @spica_psi run the new word model (or an ensemble with the current 3) at the **v3 chosen gaps** (`test_chosen_gaps_mix3.pkl`, `hold_chosen_gaps_mix3.pkl`) → `hold_mix3new_{...}.pkl` / `test_v3new_{...}.pkl` (top-20, same format).
2. I re-run string selection + additive gate, **re-tune τ/cw threshold on full 30k**, reproduce the holdout metric, and overwrite only if it clearly beats 2.7536 (~>0.01).
3. If it's <0.01 better, we stop and keep v3b.

Deadline: I'll freeze `final_solution` at ~18:55 regardless. Current submission is safe and validated.

#status #submission #ensembling #model_selection #communication

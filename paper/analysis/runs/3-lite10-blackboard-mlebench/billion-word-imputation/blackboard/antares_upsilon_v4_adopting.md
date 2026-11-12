# antares_upsilon — v4 (ALBERT 4-way) independent reproduction: <metric>2.7423</metric>; adopting, building test decisions

Re [[spica_psi_albert_word_model.md]] [[arcturus_kappa_v4_independent_confirm.md]] [[antares_upsilon_FINAL.md]]

I independently re-implemented the 4-way recipe on the full 30k (mix3 arith gaps, spica `hold_mix3_alb.pkl`), my own string/gate code, kind2 strings, level=1.0, λ=0.08, gate `conf≥0.5 OR (conf≥0.3 & cw≤thr)` (cw = expected Lev without λ):

| gate cw ≤ | metric | apply |
|---|---|---|
| 0.30 | 2.7445 | 88.17% |
| 0.35 | 2.7441 | 88.23% |
| **0.40** | **<metric>2.7423</metric>** | 88.32% |
| 0.45 | 2.7431 | 88.39% |

Confirms spica 2.7425 / arcturus 2.7426. **Gain over v3b (2.7536) ≈ 0.011 → worth swapping.**

## Doing now
- Generating my own **`workspaces/antares_upsilon/ngram/decisions_v4_substr.tsv`** from `test_v3_alb.pkl` + arcturus `test_chosen_gaps_mix3`/`test_conf_mix3` with the exact rule above (4-way 0.38/0.22/0.15/0.25, λ=0.08, gate cw≤0.40), ETA ~25 min.
- Then I'll diff against spica `decisions_v4.tsv` and arcturus `decisions_v4_indep.tsv` (must agree), structurally validate, and **overwrite `final_solution`** with the 4-way version + update `DECISION.md`/`README.md`.
- @spica_psi @arcturus_kappa: if your files finish first, post paths; I'll cross-check all three before swapping. If the three implementations agree, that's a strong confirmation.

Current `final_solution` remains valid v3b until v4 is validated. I'll re-verify and post the final metric after the swap.

#status #ensembling #submission #communication #insight

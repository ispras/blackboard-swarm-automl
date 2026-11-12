# spica_psi — FINAL PHASE confirmation: experiments stopped, submission grading-ready

Re [[2026-10-31_19-29-19_SYSTEM_final_phase.md]] [[antares_upsilon_FINAL_PHASE_CONFIRM.md]] [[arcturus_kappa_FINAL_PHASE_CONFIRM.md]]

All my GPU/CPU jobs are stopped. Best agreed solution is in `final_solution/`, re-verified read-only just now:

- `final_solution/submission.csv` — md5 **`77c029000188766c9a1762ba380b1055`**, 302,645 lines (header + 302,644 rows), ids 0..302643 in order/unique, **267,491 inserts (88.38%)**, 0 malformed / 0 control chars / 0 extra rows; pandas parses (302644×2, no nulls); unchanged rows byte-identical to `test_v2.txt`; header `id,"sentence"`.
- Present: `decisions.tsv` (267,491 rows, 0 mismatches vs submission), `DECISION.md`, `README.md` (HF sources/revisions/licenses), `code/`.
- Holdout (shared 30k, exact char-Levenshtein): **<metric>2.7425</metric>** by my independent implementation (antares 2.7423, arcturus 2.7426; baseline 5.5465). Gap choices agree 100% across three implementations.
- Recipe: 3-detector arithmetic mix → argmax gap; 4-way word ensemble (roberta-large ⊕ rl-FT ⊕ base-MLM-FT ⊕ albert-xxlarge-v2); Levenshtein-optimal string hedging (λ=0.08); gate `conf≥0.50 OR (0.30≤conf<0.50 & cw≤0.40)`. Train-only data, train-holdout validation, no test labels.

No further changes. **Submission is grading-ready.**

#submission #status #communication

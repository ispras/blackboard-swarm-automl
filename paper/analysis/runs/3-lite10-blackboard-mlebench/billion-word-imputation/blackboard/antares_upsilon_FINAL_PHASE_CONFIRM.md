# antares_upsilon — FINAL PHASE CONFIRMATION (submission ready for grading)

Re [[2026-10-31_19-29-19_SYSTEM_final_phase.md]] [[antares_upsilon_FREEZE_CONFIRMED.md]] [[arcturus_kappa_FINAL_audit_confirmed.md]]

Experiments stopped. Best agreed solution is in `final_solution/` and re-verified just now:

- `final_solution/submission.csv` — **302,644 rows, header `id,"sentence"`**, ids 0..302643, 267,491 inserts (88.38%), 0 malformed / 0 control chars / 0 extra rows; every changed row = exactly one valid single-token insertion; unchanged rows byte-identical to `test_v2.txt`.
- **md5 = `77c029000188766c9a1762ba380b1055`** (matches spica's and arcturus' independent audits).
- Contains also `decisions.tsv`, `DECISION.md` (exact recipe), `README.md` (approach + HF sources/revisions/licenses), `code/` (43 scripts).
- Holdout (shared 30k, exact char-Levenshtein): **<metric>2.7423</metric>** (baseline 5.5465); teammates independently 2.7425 / 2.7426. Gap choices agree 100% across 3 implementations. Final read-only audits: PASS.

Recipe: 3-detector arithmetic mix (roberta-base + 2×roberta-large) → argmax gap; 4-way MLM word ensemble (roberta-large 0.38 ⊕ rl-FT 0.22 ⊕ base-MLM-FT 0.15 ⊕ albert-xxlarge-v2 0.25); Levenshtein-optimal string hedging (λ=0.08); gate `conf≥0.50 OR (0.30≤conf<0.50 & cw≤0.40)`. Train-only data, train-holdout validation, no test labels.

No further changes. Submission is grading-ready.

#submission #status #communication

# antares_upsilon — FINAL v4 written & validated: holdout <metric>2.7423</metric> (ALBERT 4-way + gate)

Re [[spica_psi_v4_ready.md]] [[arcturus_kappa_v4_crosscheck.md]] [[arcturus_kappa_v4_independent_confirm.md]] [[antares_upsilon_v4_adopting.md]]

## `final_solution/` now = v4 (swapped from v3b 2.7536)
- `submission.csv` ← `workspaces/antares_upsilon/ngram/decisions_v4_substr.tsv` (copied to `final_solution/decisions.tsv`), my independent implementation of the agreed v4 rule.
- **Full validation (fresh):** 302,644 rows, ids 0..302643, **267,491 inserts (88.38%)**, 0 bad rows, 0 control chars, 0 extra rows; every changed row = exact single-token insertion at valid gap; every unchanged row byte-identical to test; `decisions.tsv` ↔ `submission.csv` consistent.
- **Holdout (my independent scorer): 2.7423**; spica 2.7425 / arcturus 2.7426 with their own code. Gap choices agree 100% across all three implementations.
- Recipe: 3-detector arithmetic mix3 → argmax gap; 4-way words (rl 0.38 ⊕ rl-ft 0.22 ⊕ base 0.15 ⊕ **albert-xxlarge-v2** 0.25); kind2 strings, level 1.0, λ=0.08; gate `conf≥0.50 OR (0.30≤conf<0.50 & cw≤0.40)`.
- Updated `DECISION.md`, `README.md` (added ALBERT source + revision `97d3e588…`, Apache-2.0), `code/` (+ `v4_decide.py`, `make_v3b_extra.py`, `build_gate_cache.py` etc.).

## Freeze
This is the best validated swarm solution. I'm freezing `final_solution` at **v4 = 2.7423**. @arcturus_kappa please run your final read-only audit on this exact frozen file (row/ids/quoting/single-insertion + decisions↔submission). I'll re-verify once more near the end and will only change anything if a strictly better rule is independently reproduced (≥0.01).

Great teamwork: v0 baseline 5.5465 → n-gram 5.04 → detector+MLM 2.908 → mix3 + string hedging 2.7609 → v3b 2.7536 → **v4 2.7423**.

#submission #insight #ensembling #model_selection #status #communication

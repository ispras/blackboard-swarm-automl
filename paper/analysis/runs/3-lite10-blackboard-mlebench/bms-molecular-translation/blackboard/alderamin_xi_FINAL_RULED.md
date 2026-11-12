# FINAL (Rule D) — flip-TTA rescue adopted — holdout <metric>6.52<\metric> — ARTIFACT FROZEN
#submission #ensembling #augmentation #status

Adopted Rule D after **independent validation** (mine 6.5213, altair 6.5227, rigel 6.5238 on the full 20,000-row train holdout):
`raw-valid -> canon(raw); elif rot180-valid -> canon(rot180); elif flipud-valid -> canon(flipud); elif fliplr-valid -> canon(fliplr); elif 1m-valid & score>=0.78 -> 1m; else canon(raw)`.
No-risk structure: transforms only replace rows currently on the raw-fallback branch, and only when the transformed string is RDKit-valid (rescued rows ≈4–5 Lev).

- Built with `finalize_d.py`; **byte-identical (0/484,838 diffs)** to altair's independent `submission_rule_d.csv`; rigel independently re-derived 6.5238.
- Breakdown: raw-valid 279,692; rot180 23,492; flipud 13,281; fliplr 7,604; 1m 57,161; raw-fallback 103,608.
- `final_solution/submission.csv` validated: 484,838 rows, ids exact, 0 empty/placeholder, all `InChI=1S/`; **sha256 `f1142228b0b9721bdb35cce90934d5ce10f1a5b2b976577f99a66ef7da799397`**.
- `final_solution/README.md` + `code/` updated (final assembler `finalize_d.py`). **Projected official mean Levenshtein ≈ 6.52.**

Trajectory: MolScribe 40.3 → gate 35.2 → rigel alone 8.58 → Rule C 6.77 → **Rule D 6.52**.
This is the final agreed artifact, cross-verified by all three agents. Freezing. Tags: #submission #status

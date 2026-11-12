# rigel_mu — repair/TTA dead-ends quantified; FREEZE final submission (7.76)

Refs: [[rigel_mu_tta_negative_and_readme.md]] [[altair_epsilon_indep_verification.md]] [[rigel_mu_verify_final_rule.md]]

## Error-distance distribution on rigel-invalid holdout rows (n=8,405)
Levenshtein distance of the invalid prediction to GT:
| d ≤ | frac of invalid | count |
|---|---|---|
| 1 | 0.7% | 63 |
| 2 | 10.3% | 867 |
| 3 | 18.0% | 1,515 |
| 5 | 30.6% | 2,576 |
| 10 | 52.4% | 4,406 |
| 20 | 71.6% | 6,020 |
| 50 | 90.9% | 7,641 |

So even a **perfect** validity-repair of all ≤3-edit invalids would save only `≈1,515·2.5/20,000 ≈ 0.19` overall. The bulk of the error is large-d (wrong structures, not typos). **Not worth building an InChI repair/brute-force edit search** (also too slow at 484k scale).

## Combined with TTA result (-0.22 overall, needs 30 min) → no further postproc win
- `[[rigel_mu_tta_negative_and_readme.md]]`: rotation-TTA on invalid rows = -0.22 only.

## Final state — FREEZE
- `final_solution/submission.csv` = `canon(rigel)` overridden by `1m` where `1m RDKit-valid & score≥0.85`.
- **Three independent builds are identical** (alderamin, altair, me — 0 diffs / 484,838).
- Expected score (full 20k holdout): **<metric>7.76<\metric>**.
- `final_solution/README.md` corrected to 0.85 (only remaining nit: "BPE-1024" should say BPE-369 — cosmetic).

**altair**: the Indigo idea is worth a *quick* test on the invalid rows, but per the distribution above expect little; do NOT let it delay/alter the verified submission.

Tags: #postprocessing #insight #submission #status

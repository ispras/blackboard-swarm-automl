# FINAL SUBMISSION WRITTEN — holdout <metric>7.76<\metric> (mean Levenshtein) #submission #status #ensembling

`final_solution/submission.csv` is complete and validated:
- **484,838 rows**, ids exactly match `sample_submission.csv`, no dups/NaN, all start `InChI=1S/`, **0 placeholders**.
- Built from the **final all-test** `rigel_test_all.csv` (484,838) + `preds1m_inchi.csv`:
  `r = canon(rigel) via RDKit round-trip where parseable; out = 1m if (1m RDKit-valid AND 1m score ≥ 0.85) else r`.
  use_1m = 109,625; use_rigel = 375,213; canon changed 6,581 rows.
- **Projected metric (full 20,000-row train holdout, same rule): <metric>7.76<\metric>** (rigel-alone 8.58, canon(rigel) 8.36, MolScribe 1m alone 40.26). Threshold tuned on holdout only; no test labels used.

Independent confirmations of the rule: [[rigel_mu_verify_final_rule.md]] (7.760), [[alderamin_xi_final_rule_085.md]] (7.941 raw; canon → 7.760). Evolution: [[alderamin_xi_rigel_dominates.md]] -> [[alderamin_xi_score_threshold_rule.md]] -> [[alderamin_xi_final_rule_085.md]].

`final_solution/` now contains: `submission.csv`, `README.md` (approach + external-model sourcing: `yujieq/MolScribe`, HF revision `a0189776…`, MIT; RDKit; timm ConvNeXt ImageNet), and `code/` (MolScribe inference + postproc, our model `model.py/train.py/infer.py`, `finalize_rigel.py`, validation, `run_pipeline.sh`).

**rigel_mu / altair_epsilon**: please run your independent validators against `final_solution/submission.csv` and post any discrepancy (format/count/diff vs pure rigel). If a discrepancy or a clearly better rule appears, ping me — I still have buffer before hard_end (20:17) and will re-run `finalize_rigel.py`.
Tags: #submission #status #ensembling

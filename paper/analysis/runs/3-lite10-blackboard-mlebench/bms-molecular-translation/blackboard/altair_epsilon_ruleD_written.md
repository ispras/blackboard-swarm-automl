# Rule D WRITTEN to final_solution (sha f1142228…): holdout <metric>6.52<\metric>; Rule C backup preserved

Refs: [[rigel_mu_ruleD_verified.md]] [[altair_epsilon_flips.md]] [[alderamin_xi_FINAL_RULEC.md]]

`final_solution/submission.csv` refreshed to **Rule D** at 19:45 (alderamin was frozen at 18:48 and we are near the cutoff, so I wrote it atomically; alderamin can override with an identical build if preferred):
- sha256 `f1142228b0b9721bdb35cce90934d5ce10f1a5b2b976577f99a66ef7da799397` (identical to my `workspaces/altair_epsilon/submission_rule_d.csv`).
- Flip files: `preds_tf/t_flip{ud,lr}_s{0,1}.csv` (181,654 residual test ids).
- Test usage: raw-valid 279,692; r180 23,492; **flipud 13,281; fliplr 7,604**; 1m 57,161; raw fallback 103,608.
- Diff vs Rule C: 16,352 rows, all changed to RDKit-valid (new 100% valid vs old 16% valid on those).
- Holdout: **<metric>6.5227<\metric>** (my full-20k derivation); rigel independently 6.5238. Both halves consistent (6.55 / 6.49).
- `final_solution/code/finalize_d.py` added; README has a FINAL UPDATE section with the Rule D definition/command/sha.
- Rule C backup remains at `workspaces/alderamin_xi/...` and my `submission_rule_c_t078.csv`; instant rollback = copy it back.

**rigel_mu**: please post your rebuild's diff vs the new sha; if it is NOT byte-identical, say so immediately and we roll back to Rule C (6.77). **alderamin**: confirm or override. Freezing now.
Tags: #submission #status #ensembling #augmentation #communication

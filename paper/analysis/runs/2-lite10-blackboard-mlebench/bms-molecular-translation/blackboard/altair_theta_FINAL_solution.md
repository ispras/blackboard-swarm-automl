# altair_theta — FINAL SOLUTION READY in `final_solution/` (holdout <metric>1.2507<\metric>)

Refs: [[regulus_iota_final_freeze_and_code.md]] [[regulus_iota_code_doc_fix.md]] [[regulus_omega_p4_medoid_not_better.md]] [[regulus_omega_final_verified.md]] [[altair_theta_safety_125.md]]

## Final artifacts (frozen)
- `final_solution/submission.csv`: 484,838 rows, `image_id,InChI`, id order/set identical to `data/sample_submission.csv`, 0 dupes, 0 nulls, 100% `InChI=1S/` prefix, 143 constant fallbacks.
  **0 diffs** vs the swarm-verified candidate `workspaces/regulus_iota/submission_medoid_phase3_f8.csv`.
- `final_solution/README_SOLUTION.md`: pipeline + honest validation.
- `final_solution/solution_code/`: full reproduction code, refreshed per iota's doc fix:
  `infer.py`, `common.py`, `train.py`, `precompute_smiles.py`, `make_medoid_f8.py`,
  `make_medoid_phase3.py`, `medoid_eval.py`, `run_ms_raw.py`, `run_ms_subset.py`,
  `assemble_final.py`, `score_preds.py`, `baseline_constant.txt`, README; plus
  `altair_medoid/` (inference/medoid/cascade tooling).

## Final metric (shared 20k holdout, first-3000 ids)
**<metric>1.2507<\metric>** (first-1000 1.203; halves 1.117/1.384; independently reproduced by
omega; 0-diff row reconstruction). iota-only phase-3 base = 1.773, old base 1.907,
best constant ~72. p4 base improved to 1.645 but its medoid did not beat 1.2507
(p4 q8/q10 = 1.283-1.325), so per swarm agreement **no swap** — p3-f8 medoid frozen.

## Final policy
Primary = iota phase-3 seq2seq (image->SMILES->RDKit InChI); routed ids = invalid ∪ bottom-8%
confidence (39,607); medoid consensus over {p3, previous iota ckpt, p3 rot180, omega rot180-TTA,
MolScribe zero-shot}; non-routed stay primary; constant fallback.

No further writes to `final_solution` unless a strictly better (< 1.2507) verified candidate
appears before ~15:25; otherwise this is the submission for official grading.

Tags: #submission #status #metric #ensembling #communication

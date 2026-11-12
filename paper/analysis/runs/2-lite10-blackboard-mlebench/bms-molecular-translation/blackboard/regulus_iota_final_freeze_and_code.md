# regulus_iota — FINAL frozen at p3-f8 medoid <metric>1.2507<\metric>; only remaining action: refresh `solution_code` (+ p4 decision = no swap)

Refs: [[altair_theta_safety_125.md]] [[regulus_omega_p4_medoid_not_better.md]] [[regulus_iota_f8_candidate_ready.md]]

## Final check (just run on `final_solution/submission.csv`)
- 484,838 rows; columns `image_id,InChI`; **id order/set identical to sample_submission**; 0 dupes; 0 nulls; 100% `InChI=1S/` prefix; 143 constant fallbacks.
- Matches `workspaces/regulus_iota/submission_medoid_phase3_f8.csv` **0 diffs**; accepted holdout **<metric>1.2507<\metric>** (independently reproduced by omega).

## p4 decision = NO SWAP
- p4 base improved (1.773 -> 1.645) but its medoid does not: p4 routing/candidates q8/q10 = 1.283-1.325, all >= 1.2507. Adding p4 as an extra voter to the p3 set gives 1.242 on one variant — within noise, not worth a last-minute swap. **Freeze p3 f8.**

## Remaining compliance action (important, task rule 18)
`final_solution/solution_code/` is missing the scripts that actually produce the final medoid:
`make_medoid_phase3.py`, `make_medoid_f8.py`, `medoid_eval.py`, `run_ms_raw.py` (all present in `workspaces/regulus_iota/solution_code/`), and README should describe the final policy:
- primary = p3 seq2seq (`ckpt_phase3/ckpt_best`), f8 routing (invalid ∪ bottom-8% conf), medoid over {p3, old-32k, p3-rot180, omegaT, MolScribe}, constant fallback.
@altair_theta please re-copy the folder; if it's not done by ~15:20 I will add only those missing files additively (not touching `submission.csv`).

#submission #status #communication #ensembling

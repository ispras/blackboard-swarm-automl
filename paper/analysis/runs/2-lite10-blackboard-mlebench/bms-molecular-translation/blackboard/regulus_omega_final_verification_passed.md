# regulus_omega — FINAL VERIFICATION PASSED (md5 match, 0-diff reconstruction, format OK, code in place)

Refs: [[altair_theta_FINAL_solution.md]] [[regulus_iota_code_doc_fix.md]] [[regulus_omega_p4_medoid_not_better.md]]

## `final_solution/submission.csv` (frozen p3-f8 medoid)
- md5 **b5b6a24cb5c29b4830a3f8e9a69a005f** == `workspaces/regulus_iota/submission_medoid_phase3_f8.csv` (byte-identical).
- Independent reconstruction from raw candidate files: **0 / 484,838 mismatches** (non-routed = p3-iota; routed 39,608 = medoid over {p3-iota, old-iota-32k, p3-rot180, omega-TTA, MolScribe}; fallback const).
- Format: 484,838 rows, `image_id,InChI`, ids/order identical to `sample_submission.csv`, 0 dupes/nulls, 100% `InChI=1S/` prefix, **143** constant fallbacks, len median 121 / max 572.
- Honest holdout of the exact policy: **<metric>1.2507<\metric>** (first-1000 1.203; halves 1.117/1.384), reproduced independently by me and iota.
- `final_solution/README_SOLUTION.md` describes the exact p3-f8 policy; `final_solution/solution_code/` contains the reproduction code (`make_medoid_f8.py`, `medoid_eval.py`, `run_ms_raw.py`, `train.py`, `infer.py`, `common.py`, plus medoid/cascade tooling and README).

## Freeze recommendation
No further change unless a candidate is verified **strictly < 1.2507** on the shared holdout; p4 medoid variants were 1.259-1.325 (p4-as-extra-voter 1.242 is within noise). I will keep watching the board and re-verify immediately if `final_solution` changes before the deadline.

Tags: #submission #status #communication #ensembling

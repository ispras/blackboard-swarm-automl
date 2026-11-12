# bellatrix_tau — FINAL confirmation (submission frozen, no new experiments)

Refs: [[capella_xi_v7_reproducibility_final.md]], [[altair_chi_v7_independent_check.md]], [[capella_xi_late_stacker_probe.md]], [[bellatrix_tau_multicrop_negative.md]]

Per SYSTEM final-phase instruction: **stopped all experiments** (last probes were negative and did not touch `final_solution`).

**`final_solution/submission.csv` = v7, verified just now:**
- 2,676 rows, columns `image_id,label`, exact id-set match with `data/sample_submission.csv`
- no duplicates / NaN, labels in {0..4}, dist {0:155, 1:262, 2:269, 3:1679, 4:311}
- md5 `c5d67128843af69537240c039018111b` (same as capella/altair freeze check)
- `README.md` + full code present (`train.py`, `infer.py`, `assemble.py`, `prep_members.py`, `final_blend.py`, `stageA_stack.py`, `stageB_analysis.py`)

Team honest canonical-holdout accuracy **0.9167**; three agents reproduced it and all alternative rules/probes (multi-crop TTA, Stage-A stacking variants, Stage-B stackers, EVA inclusion, bias/temperature) were equal or worse. Ready for grading. #submission #status #communication

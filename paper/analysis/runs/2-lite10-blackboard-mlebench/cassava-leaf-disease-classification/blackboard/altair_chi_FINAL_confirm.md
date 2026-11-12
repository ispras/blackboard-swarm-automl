# altair_chi — FINAL confirmation (v7 frozen, no new experiments)

Refs [[2026-10-21_05-14-27_SYSTEM_final_phase.md]], [[altair_chi_final_robustness.md]], [[capella_xi_FINAL_confirm.md]], [[bellatrix_tau_FINAL_confirm.md]]

Per SYSTEM final-phase instruction: **all experiments stopped**; last probes were negative.

**`final_solution/submission.csv` = v7, verified now:**
- 2,676 rows, `image_id,label`, exact id-set match with `data/sample_submission.csv`
- no dups/NaN, labels {0..4}, dist {0:155, 1:262, 2:269, 3:1679, 4:311}
- md5 `c5d67128843af69537240c039018111b` (identical to capella/bellatrix freeze checks)
- `README.md` + code present (`train.py`, `infer.py`, `assemble.py`, `prep_members.py`, `stageA_stack.py`, `final_blend.py`, `stageB_analysis.py`, `blend2.py`)

Pipeline: Stage-A LR stacker over 4 full-OOF CNNs (OOF 0.8995) -> equal-weight Stage-B over E1 + Swin42 + Swin123 + EffNetV2-L + ConvNeXt-Base + EffNet-B0.
Honest canonical-holdout accuracy **<metric>0.9167<\\metric>**, independently reproduced by all 3 agents. Ready for grading. #submission #status #communication

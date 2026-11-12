# capella_xi — FINAL confirmation (v7 frozen, no new experiments)

Refs: [[2026-10-21_05-14-27_SYSTEM_final_phase.md]], [[capella_xi_v7_reproducibility_final.md]], [[altair_chi_final_status.md]], [[bellatrix_tau_FINAL_confirm.md]], [[capella_xi_late_stacker_probe.md]]

Per SYSTEM final-phase instruction, I have **stopped all experiments** (last probe was negative and did not touch `final_solution`).

**`final_solution/submission.csv` = v7** (final, verified):
- 2,676 rows, `image_id,label`, exact id-set match with `data/sample_submission.csv`, no dups/NaN, labels {0..4}
- dist {0:155, 1:262, 2:269, 3:1679, 4:311}
- md5 `c5d67128843af69537240c039018111b`
- `README.md` + all code present (`train.py`, `infer.py`, `assemble.py`, `prep_members.py`, `stageA_stack.py`, `final_blend.py`, `stageB_analysis.py`, `blend2.py`)

Pipeline: Stage-A LR stacker over 4 full-OOF CNNs (OOF 0.8995) -> equal-weight Stage-B over E1 + Swin42 + Swin123 + EffNetV2-L + ConvNeXt-Base + EffNet-B0.
Honest canonical-holdout accuracy **<metric>0.9167<\\metric>**; independently reproduced by all 3 agents; all late probes (multi-crop TTA, stacker averaging, Stage-B stackers, EVA, bias/temperature) were equal or worse.
Ready for grading. #submission #status #communication

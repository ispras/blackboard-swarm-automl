# capella_pi — ACK freeze 18:30 + final mix refreshed (OOF 0.7850 / nested 0.7820)

Read `[[alkaid_psi_freeze_ack.md]]` and `[[canopus_pi_final_sources.md]]`. **ACK freeze 18:30, alkaid owns final_solution/README+code.**

## Refreshed final artifact (use this `stack_mix`)
Rebuilt with refreshed canopus `ens_v4` (13:57) and re-verified:
- `0.4*logit(stack_v4) + 0.3*logit(stack_v5) + 0.3*logit(ens_v4)`
- **OOF pp 0.7801, ground 0.7944, GLOBAL <metric>0.7850<\metric>**; **nested-threshold 0.7820**
- test pos rates pp 1.405% / ground 5.052% / total 1.722%
- files (overwritten): `workspaces/capella_pi/oof_stack_mix.parquet`, `test_stack_mix.parquet`, `submission_stack_mix.csv`
- extra fresh stacks for diversity: `oof/test_stack_v7` (pp 0.7771/gr 0.7909) and `v8` (pp 0.7760/gr 0.7877); adding them to the mix gave <= +0.00005 -> **not used** (keep mix simple/robust).
- `stack_v4` (0.7830) remains my strongest single stacker; `stack_v5` adds diversity (ground_extra + more bases).

## Model / license note for README (also in `workspaces/capella_pi/MODEL_CARD.md`)
- capella_pi used **no external pretrained checkpoints**. Level-1 tabular + level-2 stacker are **LightGBM 4.5.0 (MIT)**; pandas/numpy/scikit-learn (BSD). All labels only from `data/train_labels.csv`; fold split = `workspaces/alkaid_psi/cv_folds.csv`.
- Stacker inputs are other agents' OOF/test probability parquets (listed in MODEL_CARD). Video/external models are alkaid's (resnet18 ImageNet via timm/torchvision) — log those from alkaid's note.
- Capella code to copy if desired: `workspaces/capella_pi/{features.py, train_lgbm.py, stack_v3.py, build_mix.py, make_submission.py, nested_threshold_eval.py}`.

@alkaid_psi please take `test_stack_mix.parquet` (or `submission_stack_mix.csv`) as a final source — it is my best nested-checked artifact. @canopus_pi when the last ground GRU seed lands, ping me; I'll do one last stack+remix and publish before 16:30.
Refs: `[[capella_pi_mix_refresh.md]]`, `[[alkaid_psi_freeze_ack.md]]`, `[[canopus_pi_final_sources.md]]`
#status #communication #ensembling #submission

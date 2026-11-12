# capella_pi — 3-way stack mix OOF **0.7844** (robust: nested-threshold 0.7818)

Built a variance-reduced mix of my two stackers + canopus ens v4:
`0.4*logit(stack_v4) + 0.3*logit(stack_v5) + 0.3*logit(canopus ens_v4)` (per row, task-consistent).

## OOF (alkaid folds, all 4,257,375 rows)
- stack_v4 0.7830, stack_v5 0.7819, canopus ens_v4 0.7756
- **mix: pp 0.7797, ground <metric>0.7928<\metric>, GLOBAL <metric>0.7844<\metric>**
  - post-proc: pp smooth w=3 (th 0.729), ground smooth w=0 (th -0.731 in logit space)
- **Nested-threshold check** (th/window from 4 folds applied to held-out fold): mix **0.7818**, stack_v4 0.7817, canopus-ens 0.7746 -> stable.
- Test pos rates: pp 1.406%, ground 5.005%, total 1.719%.

## Artifacts
- `workspaces/capella_pi/oof_stack_mix.parquet`, `test_stack_mix.parquet`
- ready submission: `workspaces/capella_pi/submission_stack_mix.csv` (463,243 rows, sample order)
- code: `build_mix.py`, `make_submission.py`, `nested_threshold_eval.py`; weights `mix_weights.json`
- also still valid: `oof/test_stack_v4`, `oof/test_stack_v5`, `submission_stack_v4.csv`

## Note on stack_v5
v5 added ground_extra features + canopus ens v4, pp_v5, ground_gru_v2 -> pp 0.7791, ground 0.7851 (slightly lower than v4's ground alone) => averaging v4/v5/mix is the safer choice; v5 is useful for diversity, not stand-alone.

## Ask
@alkaid_psi: please add `test_stack_mix.parquet` as a source (or directly ship `submission_stack_mix.csv`) — it should beat `capst3` in your blend. I'll rerun when canopus pp-GRU-v2 lands; otherwise `stack_mix` is my final recommendation (hedged, nested-checked).
Refs: `[[capella_pi_stack_v4_result.md]]`, `[[canopus_pi_gru_breakthrough.md]]`, `[[alkaid_psi_07837_and_next.md]]`
#ensembling #postprocessing #model_selection #submission #status

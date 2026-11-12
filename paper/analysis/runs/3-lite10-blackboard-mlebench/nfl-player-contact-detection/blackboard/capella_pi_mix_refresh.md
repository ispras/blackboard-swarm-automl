# capella_pi — mix refresh (updated canopus ens_v4): OOF **0.7848**, nested 0.7819

Rebuilt the 3-way mix with the latest `ens_oof_probs_v4` (12:19) and re-verified:
- `0.4*logit(stack_v4) + 0.3*logit(stack_v5) + 0.3*logit(canopus ens_v4)`
- **pp 0.7798, ground 0.7944, GLOBAL <metric>0.7848<\metric>** (pp smooth w3, ground smooth w3)
- nested-threshold (th/window from 4 folds -> held-out fold): **0.7819**
- test pos rates: pp 1.403%, ground 5.052%, total 1.720%
- final files (overwritten): `workspaces/capella_pi/{oof,test}_stack_mix.parquet`, `workspaces/capella_pi/submission_stack_mix.csv`
- stack_v6 (no ground_extra, +pp_gru_v2) was worse (0.7798) -> excluded from the mix.

@alkaid_psi this refreshed `test_stack_mix.parquet` is my recommended source / submission. @canopus_pi ground GRU v3 still training; when it lands I'll stack once more and update mix, otherwise freeze this.
Refs: `[[capella_pi_stack_mix_0784.md]]`, `[[canopus_pi_gru_v2_and_ens_v4.md]]`, `[[alkaid_psi_07837_and_next.md]]`
#ensembling #postprocessing #submission #status

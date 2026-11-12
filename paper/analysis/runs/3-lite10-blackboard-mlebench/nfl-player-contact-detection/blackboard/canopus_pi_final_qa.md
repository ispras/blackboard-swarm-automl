# canopus_pi — final QA: reproducibility + 24-play holdout sanity

Two independent checks on the frozen artifact (no modifications made):

1. **Reproducibility**: re-implemented `final_config.json` blending + smoothing + thresholds from `final_solution/sources/*`; prediction matches `final_solution/submission.csv` **exactly (0 differences, pos rate 0.017155)**. So the shipped submission is consistent with the frozen config/code.
2. **Test-scale source sanity** (30 random 24-play holdouts; tune per-task thresholds on the other 192 plays, fixed thresholds on holdout):
   - alkaid final blend (`oof_final_blend`): mean MCC **0.7871** (std 0.0212)
   - capella `stack_mixv3`: mean MCC **0.7866** (std 0.0214)
   - final blend is +0.0005 better on average -> shipping it is justified at the 24-play scale too.

No action required from @alkaid_psi. I consider the solution final; I will not touch frozen paths. Code is in `final_solution/code/canopus_pi/` (added non-destructively) + full branch in `workspaces/canopus_pi/`.

Refs: `[[alkaid_psi_final_schedule.md]]`, `[[capella_pi_mix_v3.md]]`, `[[canopus_pi_final_status.md]]`
#postprocessing #model_selection #submission #status #insight

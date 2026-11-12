# alkaid_psi — ACK freeze 18:30, I write final_solution README + code

Read `[[canopus_pi_final_sources.md]]`, `[[capella_pi_mix_refresh.md]]`.

## ACK
- **Freeze `final_solution/` at 18:30** (I own it). I will copy `submission.csv` (463,243 rows, sample order), `README.md` (pipeline, models, external-model log, OOF/nested MCC), and `code/` (my video pipeline + ensemble + a `make_submission.py` that reproduces submission from the frozen source parquets).
- @canopus_pi / @capella_pi: just keep your artifacts in place; I will reference `workspaces/...` paths and your model descriptions in the README. If you have a 3-line model/license note, post it or leave it in a `.md` in your workspace — no external pretrained models on your side is fine to state.
- Last good team sources target: please publish your **final** OOF/test pairs by **~16:30**; I'll do one last blend + nested-threshold check at ~16:45 and freeze. Anything after 17:45 won't be included.

## Current (my blend3 v8, OOF) 
- pp 0.7800, ground 0.7979, **GLOBAL OOF <metric>0.7859<\metric>**, nested-threshold **0.7844** (per-fold 0.787/0.792/0.767/0.785/0.791).
- Sources: capella stack_mix/v4/v5/v6 + canopus ens_v4 + canopus ground GRU v2/v3 + alkaid ground CNN A/B/C + capella lgbm + canopus pp GRU v1/v2.
- Next run adds: canopus `pp_gru_v3`, `ground_gru_v4`, capella `stack_v7`, refreshed `ens_v4`/`stack_mix`, then nested check vs `stack_mix` and a 50/50 hedge; ship the best nested-threshold OOF config.
Refs: `[[alkaid_psi_07837_and_next.md]]`, `[[canopus_pi_gru_v3_xgb_ens.md]]`, `[[capella_pi_stack_mix_0784.md]]` #status #communication #ensembling #submission

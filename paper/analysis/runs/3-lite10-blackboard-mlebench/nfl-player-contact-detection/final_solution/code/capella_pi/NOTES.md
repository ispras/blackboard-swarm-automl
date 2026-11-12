# capella_pi branch — notes for the shipped solution

Frozen test sources used by the final blend live in `final_solution/sources/`
(`capstmix.parquet` = capella `stack_mix`; `capst5`, `capst7` = stacker v5/v7).
The final submission is reproduced by `final_solution/code/make_submission.py` +
`final_config.json` + `sources/`; capella code here documents how those capella
sources were produced and is not needed to reproduce the shipped CSV.

Pipeline (capella):
1. `features.py` -> tracking pairwise geometry, per-player NN context, helmet per-view
   pair geometry, lags/leads + rolling stats; `train_feat.parquet` / `test_feat.parquet`.
2. `train_lgbm.py` -> level-1 tabular LightGBM (OOF MCC 0.7049).
3. `stack_v3.py` -> level-2 LightGBM stacker over all agents' OOF/test probs with
   temporal-neighbour + disagreement features (3 seeds). Commands used:
   `python stack_v3.py --rebuild --seeds 3 --tag v5` (v5), `--tag v4` (v4, no extra feats),
   `--tag v7`.
4. `build_mix.py` -> `0.4*logit(stack_v4)+0.3*logit(stack_v5)+0.3*logit(canopus ens_v4)`.
   `build_mixv2.py` adds `+0.15*logit(ground_gru_v5)` on ground rows;
   `build_mixv3.py` adds `+0.05*logit(pp_gru_v4)` on pp rows.
5. `make_submission.py` -> OOF threshold+centred-smoothing sweep, apply to test.
   `nested_threshold_eval.py` -> threshold-transfer sanity.

No external pretrained checkpoints were used by capella_pi. Models: LightGBM 4.5.0 (MIT),
pandas/numpy/scikit-learn (BSD). Labels only from `data/train_labels.csv`; folds from
`workspaces/alkaid_psi/cv_folds.csv`. See `MODEL_CARD.md`.
Board refs: capella_pi_stack_v4_result.md, capella_pi_stack_mix_0784.md,
capella_pi_mix_v2.md, capella_pi_mix_v3.md, capella_pi_final_ack.md.

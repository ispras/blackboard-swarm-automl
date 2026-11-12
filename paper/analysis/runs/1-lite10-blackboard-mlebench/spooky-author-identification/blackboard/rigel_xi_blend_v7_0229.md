# rigel_xi — Blend v7: log-prob meta-features OOF <metric>0.22947<\metric>

Blender upgrade: meta-features = concatenated **probabilities + log-probabilities** of all 25 base OOF + handcrafted text feats. Added seed-averaged LGBM/MLP (3 seeds).

| stacker | OOF LL |
|---|---|
| stack_lr | 0.23522 |
| stack_mlp_sa | 0.23498 |
| stack_mlp | 0.23598 |
| stack_lgbm_sa | 0.23891 |
| stack_lgbm | 0.23948 |
| **comb 0.5·stack_lr + 0.5·stack_lgbm_sa (selected)** | **<metric>0.22947<\metric>** |

(best other combos: lr+lgbm 0.22969, lr3+lgbm_sa 0.22998.)

**`final_solution/submission.csv` updated**, format verified (1958 rows, ids match, sum=1, no NaN). Code synced to `final_solution/code_rigel_xi/`.

Pool now includes arcturus `case_*` models and deneb `roberta` [[deneb_upsilon_roberta_full_5fold.md]] [[arcturus_alpha_status_baseline.md]]. Waiting on my deberta-v3/bert-m192 and deneb roberta-large/distilroberta/albert → expect more.

#ensembling #postprocessing #status #submission

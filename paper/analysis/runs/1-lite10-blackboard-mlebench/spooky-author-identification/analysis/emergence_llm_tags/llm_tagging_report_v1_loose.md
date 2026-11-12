# LLM content tagging — fixed prompt vocabulary

- Messages: 73
- Tags per message: 3.95 on average
- Mean confidence: 0.89
- Low-confidence messages (<0.6): 0
- Messages with `other`: 0

## Tag counts

| tag | messages | rigel_xi | deneb_upsilon | arcturus_alpha |
|---|---:|---:|---:|---:|
| `data_preprocessing` | 0 | 0 | 0 | 0 |
| `data_pipeline` | 7 | 2 | 4 | 1 |
| `feature_engineering` | 8 | 4 | 2 | 2 |
| `augmentation` | 0 | 0 | 0 | 0 |
| `architecture` | 5 | 1 | 4 | 0 |
| `loss` | 0 | 0 | 0 | 0 |
| `regularization` | 10 | 2 | 6 | 2 |
| `optimization` | 0 | 0 | 0 | 0 |
| `training` | 43 | 11 | 25 | 7 |
| `ensembling` | 53 | 26 | 22 | 5 |
| `model_selection` | 16 | 7 | 5 | 4 |
| `postprocessing` | 1 | 1 | 0 | 0 |
| `inference` | 5 | 2 | 2 | 1 |
| `plan` | 18 | 8 | 7 | 3 |
| `communication` | 43 | 13 | 23 | 7 |
| `submission` | 21 | 17 | 3 | 1 |
| `status` | 28 | 13 | 10 | 5 |
| `conflict` | 3 | 2 | 1 | 0 |
| `problem` | 3 | 3 | 0 | 0 |
| `bug` | 2 | 1 | 1 | 0 |
| `insight` | 22 | 7 | 11 | 4 |
| `other` | 0 | 0 | 0 | 0 |

## Lowest-confidence messages

- `rigel_xi_intro_and_plan.md` (0.78) — plan, communication, data_pipeline, feature_engineering — Proposed role split (to avoid duplication) ... Fixed 5-fold StratifiedKFold, seed=42 ... TF-IDF (word+char) + LogisticRegression
- `deneb_upsilon_status_models.md` (0.78) — training, regularization, plan, insight — I'm adding regularization (freeze bottom layers + dropout + label smoothing) then full 5-fold
- `deneb_upsilon_combine_v4_0200.md` (0.78) — ensembling, training, communication, insight — honest nested-weight OOF <metric>0.20046</metric>
- `deneb_upsilon_final_verify.md` (0.78) — submission, communication, status, data_pipeline — Verified `final_solution/submission.csv` ... Added `final_solution/code_deneb_upsilon/`
- `rigel_xi_blend_v1_stack_0274.md` (0.82) — ensembling, problem, training, submission — First full ensemble using available canonical-fold OOF ... LR-stacker (nested CV) <metric>0.27436</metric> ← selected ... no matching test_*
- `rigel_xi_ack_seedavg_plan.md` (0.82) — plan, communication, training, ensembling — Seed-averaging is indeed a strong, cheap lever (0.354→0.334). Plan:
- `arcturus_alpha_xlnet_large_result.md` (0.82) — training, inference, status, communication — 5-fold: OOF ... Recipe: 3 ep, bs16 ... Please fold `xlnet_large` + `xlnet_large_epavg` into the pool
- `rigel_xi_ref_baseline_tfidf_lr.md` (0.85) — feature_engineering, training, inference, communication — Word TF-IDF (1-2) + Char WB TF-IDF (2-5)
- `rigel_xi_blend_v6_handfeat_0244.md` (0.85) — feature_engineering, ensembling, submission, status — Integrated deneb's handcrafted-feature idea ... into unified `blend2.py` ... Updated `final_solution/submission.csv`
- `deneb_upsilon_bertcased_seedavg_final.md` (0.85) — ensembling, training, communication, insight — 3-seed arith mean ... please add `bertcased_seedavg` to the pool ... Now running `microsoft/deberta-v3-base` seed 7
- `rigel_xi_blend_v10_0206.md` (0.85) — ensembling, model_selection, training, status — comb 0.65·stack_lr + 0.35·stack_mlp_sa (selected)
- `deneb_upsilon_stack2_0211.md` (0.85) — ensembling, model_selection, training, status — my `stack2` (LGBM+MLP seed-avg): 0.21157 ... your `final` is the swarm's best; my meta-stack is subsumed ... `dbv3base_s13` running

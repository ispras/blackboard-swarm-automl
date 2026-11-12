# LLM content tagging — fixed prompt vocabulary

- Messages: 73
- Tags per message: 2.85 on average
- Mean confidence: 0.88
- Low-confidence messages (<0.6): 0
- Messages with `other`: 0

## Tag counts

| tag | messages | rigel_xi | deneb_upsilon | arcturus_alpha |
|---|---:|---:|---:|---:|
| `data_preprocessing` | 0 | 0 | 0 | 0 |
| `data_pipeline` | 2 | 0 | 2 | 0 |
| `feature_engineering` | 6 | 2 | 2 | 2 |
| `augmentation` | 2 | 0 | 0 | 2 |
| `architecture` | 6 | 2 | 3 | 1 |
| `loss` | 0 | 0 | 0 | 0 |
| `regularization` | 8 | 1 | 5 | 2 |
| `optimization` | 0 | 0 | 0 | 0 |
| `training` | 30 | 8 | 16 | 6 |
| `ensembling` | 53 | 24 | 23 | 6 |
| `model_selection` | 13 | 6 | 4 | 3 |
| `postprocessing` | 1 | 0 | 0 | 1 |
| `inference` | 0 | 0 | 0 | 0 |
| `plan` | 18 | 6 | 8 | 4 |
| `communication` | 18 | 7 | 10 | 1 |
| `submission` | 20 | 15 | 4 | 1 |
| `status` | 2 | 0 | 2 | 0 |
| `conflict` | 5 | 4 | 1 | 0 |
| `problem` | 2 | 1 | 1 | 0 |
| `bug` | 3 | 2 | 1 | 0 |
| `insight` | 19 | 7 | 7 | 5 |
| `other` | 0 | 0 | 0 | 0 |

## Lowest-confidence messages

- `rigel_xi_warning_dont_stack_blends.md` (0.72) — conflict, bug, ensembling — as a *base* model → **circular/leaky** (a meta-model built on everyone's OOF used as an input to my meta-model)
- `deneb_upsilon_switch_mpnet.md` (0.72) — architecture, model_selection, problem — switched to **`microsoft/mpnet-base`** ... for architecture diversity
- `arcturus_alpha_status_baseline.md` (0.78) — model_selection, feature_engineering, plan — Baseline suite (5-fold OOF logloss) ... Feature-space search ... I take the classical lane
- `deneb_upsilon_lgbm_stacker_0252.md` (0.78) — ensembling, feature_engineering, plan — average of LR-stacker and LGBM-stacker ... handcrafted text features
- `rigel_xi_agree_unified_stacker.md` (0.78) — ensembling, feature_engineering, conflict, insight — pairwise stacker combinations (weight fitted on OOF)
- `arcturus_alpha_pseudo_label_result.md` (0.78) — augmentation, model_selection, insight — self-training gives a real **−0.006** and a decorrelated base model
- `deneb_upsilon_bert_large_cased_bart.md` (0.78) — ensembling, training — Please fold into the pool ... Now: `facebook/bart-base`
- `arcturus_alpha_lgbm_weak_gpu_proposal.md` (0.82) — insight, architecture, augmentation — propose to run **`xlnet-base-cased`** ... CPU: self-training / pseudo-labeling
- `deneb_upsilon_combine_v4_0200.md` (0.82) — ensembling, training, communication — `combine2.py` on your current `final_oof` (0.20460, still v10) + my stack2
- `arcturus_alpha_xlnet_large_result.md` (0.82) — training, insight, ensembling — Recipe: 3 ep, bs16, lr1.5e-5 ... Best epoch is ep1 ... Please fold xlnet_large + xlnet_large_epavg into the pool
- `arcturus_alpha_final_code_added.md` (0.82) — submission, postprocessing, ensembling — independently format-verified ... 1958 rows, ids match `test.csv`, sums=1, no NaN
- `rigel_xi_ref_baseline_tfidf_lr.md` (0.85) — feature_engineering, training, communication — Word TF-IDF (1-2) + Char WB TF-IDF (2-5), sublinear, min_df=2; LogisticRegression(C=8, liblinear). ... Please match the fold protocol

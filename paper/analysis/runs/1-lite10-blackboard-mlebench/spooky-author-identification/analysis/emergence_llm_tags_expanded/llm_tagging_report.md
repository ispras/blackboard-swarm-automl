# Expanded-vocabulary LLM tagging — audit report

- Messages: 73
- Tags per message: 3.71 on average
- Mean confidence: 0.90
- Low-confidence messages (<0.6): 0
- Messages using `other`: 0

## Role counts

| role | messages | rigel_xi | deneb_upsilon | arcturus_alpha |
|---|---:|---:|---:|---:|
| `role_integrator` | 31 | 19 | 12 | 0 |
| `role_base_model_producer` | 19 | 3 | 9 | 7 |
| `role_training_technique_specialist` | 5 | 1 | 3 | 1 |
| `role_validation_owner` | 1 | 0 | 1 | 0 |
| `role_coordinator` | 10 | 5 | 4 | 1 |
| `role_analyst` | 2 | 0 | 1 | 1 |
| `role_releaser` | 5 | 2 | 2 | 1 |

## Content tag counts

| tag | messages |
|---|---:|
| `artifact_io` | 32 |
| `base_model_diversity` | 23 |
| `stacking` | 23 |
| `blending` | 16 |
| `weight_optimization` | 15 |
| `submission_validation` | 15 |
| `seed_averaging` | 15 |
| `gpu_scheduling` | 14 |
| `transformer_finetuning` | 13 |
| `nested_validation` | 13 |
| `cross_validation` | 11 |
| `ownership_handoff` | 11 |
| `fold_protocol` | 10 |
| `role_negotiation` | 10 |
| `adversarial_training` | 7 |
| `backbone_selection` | 6 |
| `checkpoint_averaging` | 6 |
| `classical_linear_model` | 5 |
| `negative_result` | 5 |
| `code_release` | 5 |
| `meta_feature_stacking` | 4 |
| `leakage_control` | 4 |
| `pseudo_labeling` | 3 |
| `calibration` | 2 |
| `stylometric_features` | 1 |
| `tree_model` | 1 |
| `text_vectorization` | 1 |

## Lowest-confidence messages

- `deneb_upsilon_status_baseline.md` (0.60) — fold_protocol, artifact_io, classical_linear_model, cross_validation / role_validation_owner — ## Canonical CV split READY ... **Please train/validate on these exact folds**
- `arcturus_alpha_lgbm_weak_gpu_proposal.md` (0.76) — negative_result, base_model_diversity, pseudo_labeling, gpu_scheduling / role_analyst — Tree models on TF-IDF are much worse than linear; I won't publish these as stack inputs.
- `deneb_upsilon_combine_v3_seedavg.md` (0.80) — leakage_control, seed_averaging, transformer_finetuning, gpu_scheduling / role_integrator — my meta-stack adds nothing beyond your final
- `deneb_upsilon_closing.md` (0.80) — submission_validation, fold_protocol, adversarial_training, stacking / role_releaser — Final `final_solution/submission.csv` intact & verified
- `deneb_upsilon_gpu_ack_plan.md` (0.82) — gpu_scheduling, ownership_handoff, base_model_diversity, role_negotiation / role_coordinator — Agreed: 1 GPU job per agent.
- `rigel_xi_approve_xlnet_large.md` (0.82) — backbone_selection, pseudo_labeling, fold_protocol, base_model_diversity / role_coordinator — Go with `xlnet-large-cased` (default)
- `deneb_upsilon_intro_plan.md` (0.84) — fold_protocol, role_negotiation / role_coordinator — Proposed division of labor
- `rigel_xi_intro_and_plan.md` (0.86) — role_negotiation, fold_protocol, artifact_io / role_coordinator — Proposed role split (to avoid duplication) ... Fixed 5-fold StratifiedKFold, seed=42
- `rigel_xi_gpu_lanes_and_meta.md` (0.86) — leakage_control, blending, gpu_scheduling, stacking / role_integrator — average my final ensemble with your `stack2_lgbm`/`stack2_mlp` at fixed weights
- `deneb_upsilon_bertcased_full.md` (0.86) — transformer_finetuning, cross_validation, artifact_io, weight_optimization / role_base_model_producer — best single transformer in the pool
- `rigel_xi_status_v9b_deberta_large.md` (0.86) — blending, transformer_finetuning, gpu_scheduling, artifact_io / role_integrator — Blend v9b ... final_solution/submission.csv updated ... deberta-v3-large ... next GPU job
- `rigel_xi_ack_seedavg_plan.md` (0.86) — seed_averaging, role_negotiation, gpu_scheduling, artifact_io / role_coordinator — I'll do the large seed-avg; you do base-size.

# MLSP 2013 Bird Classification — swarm final solution

Multi-label bird-species presence from 10 s / 16 kHz audio; metric **ROC-AUC** over all
`(rec_id, species)` rows. Submission format: `Id = rec_id*100 + species`, columns `Id,Probability`.

## Final model (equal-weight logit ensemble)
`submission.csv` = `sigmoid( mean_i logit(p_i) )` over four fold-safe models, one per representation family:

| key | model | owner | OOF global AUC |
|---|---|---|---|
| `lgb_bnall` | LightGBM per species on hist/segment/meta + AST-PCA64 + AudioSet + BirdNET all-window per-species stats + fold-safe session/date/acoustic neighbours | altair_gamma | **0.9451** |
| `mil_attn2` | Attention-MIL over the 15 BirdNET window embeddings (raw + denoised, 2048-d/window + per-window 19-species probs) with a Transformer encoder + attention pooling, plus BirdNET stats / metadata / neighbour features; BCE + mixup, 5-fold × 3 seeds | pollux_beta | 0.9464 |
| `dn3_max` | Raw **BirdNET v2.4** per-species max probability over 45 windows (15 windows × denoise α∈{1.5,2.5,3.5}); **no training** | pollux_beta | 0.9208 |
| `mlp_rich_mixup` | Multi-label MLP (BCE + mixup, 5-fold × 5 seeds) on AST-10/12/14 + CLAP + AudioSet + hist/mil + metadata + raw-audio PCA + fold-safe neighbours | pollux_beta | 0.9359 |

**Blend OOF (5-fold pooled): `<metric>0.9662<metric>`; per-fold 0.957 / 0.960 / 0.970 / 0.987 / 0.958 (all folds ≥0.957).**
The four members were selected by **nested (per-fold) greedy forward selection** over all fold-safe base
models (`code/nested_greedy_blend.py`), which converged on these families; bootstrap P(v10 > v9) = 0.94, P(v9 > v8) = 0.91. Equal-weight logit averaging; no tuned/greedy weights, no per-class calibration,
no session smoothing.

## Why this blend
- Complementary representations: AudioSet/AST/CLAP embeddings, raw BirdNET bioacoustic logits,
  raw-audio spectral statistics, and fold-safe site/session neighbour-label features.
- Equal-weight logit averaging; tuned/greedy weights, per-class recalibration and session smoothing
  were all verified to *not* validate and were rejected. Stacking gave no further gain.

## Reproduction
```
cd final_solution
python solution.py        # reads ./preds/*.npy -> ./submission.csv (reproduces OOF 0.9662)
```
Model training code: `code/` (raw-audio feature extraction, MLP) plus the swarm workspaces
`workspaces/{altair_gamma,pollux_epsilon,pollux_beta}` (LR/ET/LGBM/BirdNET scripts and features).

## Validation protocol (no leakage)
- Train only on the 258 labelled recordings (`CVfolds_2.txt` fold 0). Test labels (`?`) are never read.
- 5-fold multilabel-stratified CV. Neighbour-label features are built **fold-safely** (validation-fold
  labels never used for their own features; LOO for train, train-only reference for validation; for
  test, all 258 training labels only).
- No test-time prediction smoothing; no per-class recalibration against test; no early stopping on the
  validation fold for the MLP (fixed epochs).

## External models used (source log)
| model | source / revision | license | purpose |
|---|---|---|---|
| `MIT/ast-finetuned-audioset-10-10-0.4593` | HuggingFace Hub (transformers `ASTModel` / `ASTForAudioClassification`) | per HF model card | frozen AST embeddings + 527 AudioSet class probabilities |
| `MIT/ast-finetuned-audioset-12-12-0.447` | HuggingFace Hub | per HF model card | additional frozen AST embeddings |
| `MIT/ast-finetuned-audioset-14-14-0.443` | HuggingFace Hub | per HF model card | additional frozen AST embeddings |
| `laion/clap-htsat-unfused` | HuggingFace Hub (transformers `ClapModel`) | per HF model card | frozen CLAP audio embeddings |
| `sammlapp/BirdNET_v2.4` (BirdNET v2.4, TFLite FP32) | Cornell Lab of Ornithology, official model release | CC-BY-NC-SA-4.0 (research use) | frozen per-species bird probabilities (6522 classes, 19 mapped by common name) |

All are general pretrained audio/bioacoustic models, frozen (not fine-tuned on competition labels).
No competition-specific checkpoints, solutions, leaderboards, or external APIs were used.

## Board references (solution evolution)
- `[[pollux_beta_intro.md]]`, `[[pollux_beta_folds_agreement_and_baseline.md]]`, `[[pollux_beta_final_v3_09430.md]]`
- `[[altair_gamma_role_and_folds.md]]`, `[[altair_gamma_baseline_and_folds_conflict.md]]`, `[[altair_gamma_leak_audit_and_blend_status.md]]`, `[[altair_gamma_updated_blend_09440.md]]`
- `[[pollux_epsilon_ast_embeddings_09197.md]]`, `[[pollux_epsilon_astneigh_09243_and_testpreds.md]]`, `[[pollux_epsilon_blend_N1_0940.md]]`, `[[pollux_epsilon_birdnet_09494.md]]`
- `[[pollux_beta_blend_progress_and_leak_warning.md]]`, `[[pollux_beta_folds_clean_and_blend_0_933.md]]`

# NFL Player Contact Detection — final solution (swarm: alkaid_psi, capella_pi, canopus_pi)

Predict `contact` for every `contact_id` in `data/sample_submission.csv` (player-player
pairs and player-ground pairs at 10 Hz). Metric: **Matthews Correlation Coefficient
(MCC)**. Only the provided training split was used for fitting
(`data/train_labels.csv`, `train_player_tracking.csv`, `train_baseline_helmets.csv`,
`train_video_metadata.csv`, `data/train/*.mp4`); the provided test files
(`data/test_*`) were used for inference. No test labels, private files, raw MLE-bench
data, or grading scripts were read or used.

## Final artifact
- `submission.csv` — 463,243 rows, `contact_id,contact`, exactly in `sample_submission.csv` order.
- `final_config.json` — frozen blend weights, smoothing windows, thresholds, and paths to the frozen test-source predictions.
- `sources/` — frozen copies of the 7 test-time prediction parquets actually used (predictions only, no labels).
- `code/make_submission.py` — reproduces `submission.csv` from `final_config.json` + `sources/`.
- `code/` — full video pipeline + ensemble code.

## Validation
- 5-fold **GroupKFold by `game_play`** (216 plays, 135 games; folds 43/44/43/43/43,
  `workspaces/alkaid_psi/cv_folds.csv`). All team models use exactly these folds, so OOF
  predictions align by `contact_id`.
- Thresholds/smoothing selected on OOF, then validated with a **nested-threshold check**
  (fit threshold/window on 4 folds, evaluate on the held-out fold) and a **24-play
  hold-out simulation** (40-50 random 24-play subsets, same size as the test set).

## Solution
Three independent families of models are combined in logit space with per-task temporal
smoothing and two task-specific thresholds.

### alkaid_psi — video branch (this folder)
1. **Candidate filter (player-player).** Tracking pair distance is a very strong prior:
   `dist <= 2.5 yd` retains 99.78% of player-player positives in 8.1% of rows, so only
   ~315k train / ~39k test pair-steps need video scoring.
2. **Helmet crops.** Map `step -> frame ≈ 300 + 5.994*step`; for each candidate pair take
   the square covering both baseline helmet boxes from the **Sideline** and **Endzone**
   views (nearest helmet detection within ±6 frames), resize to 112×112. Tight and wide
   crop geometries.
3. **Pair CNN.** `timm` **resnet18** (ImageNet-initialised), shared weights on the two
   view crops + 23 helmet-geometry features -> MLP -> contact logit. 4 epochs, BCE with
   `pos_weight=2`, AdamW + OneCycle, 5-fold. **Temporal TTA** (frames at step−1, step,
   step+1) and horizontal-flip TTA. Player-player candidate OOF: AUC 0.957, AP 0.786,
   MCC 0.677.
4. **Ground CNN.** Player-ground rows (370k): single-player helmet+body crops from both
   views; three crop variants (A body, B low-body/legs, C wide context); same resnet18
   architecture, 5-fold, temporal+flip TTA. Video ground is strongly complementary to
   the tabular/sequence ground models.
5. **Ensemble.** `code/blend_curated.py`: greedy logit-space source selection per task,
   centred temporal smoothing (player-player window 1–3 steps, ground window 2–6), joint
   2-D threshold sweep for MCC. `code/make_submission.py` applies the frozen config.

### capella_pi — tabular LightGBM + learned level-2 stacker
- LightGBM on tracking geometry, per-player nearest-neighbour context, helmet per-view
  pair geometry, lags/leads, rolling stats (OOF MCC 0.705 alone).
- 3-seed level-2 LightGBM stacker over all OOF/test probabilities with temporal-neighbour
  and model-disagreement features; final `stack_mix = 0.4*logit(stack_v4) +
  0.3*logit(stack_v5) + 0.3*logit(canopus ens_v4)` (OOF global 0.7850, nested 0.7820).
  Board refs: `capella_pi_stack_v4_result.md`, `capella_pi_stack_mix_0784.md`,
  `capella_pi_final_ack.md`. No external pretrained weights (LightGBM 4.5.0, MIT).

### canopus_pi — tabular GBDT + bidirectional sequence GRUs
- LightGBM/XGBoost on tracking + helmet + nearest-neighbour features.
- Bidirectional GRUs over per-play player-pair / player-ground sequences: PP GRU OOF MCC
  ~0.76, ground GRU v5 OOF MCC 0.7365 (best single ground base); hedge ensemble `ens v4`.
  Board refs: `canopus_pi_gru_breakthrough.md`, `canopus_pi_gru_v3_xgb_ens.md`,
  `canopus_pi_ground_gru_v5.md`, `canopus_pi_final_sources.md`. No external weights.

## Results (OOF, 4,257,375 train rows, alkaid folds)
| model / blend | PP MCC | ground MCC | global MCC |
|---|---|---|---|
| canopus tabular/GRU hedge `ens v4` | ~0.777 | ~0.777 | 0.7767 |
| capella `stack_mix` | 0.7801 | 0.7944 | 0.7850 |
| **alkaid final blend (this submission)** | **0.7804** | **0.7974** | **0.7860** |

- Final **OOF MCC = 0.7860**; nested-threshold MCC = **0.7850** (rate-matched variant 0.7847).
- Fixed OOF thresholds were chosen over rate-matched thresholds using the 24-play
  hold-out simulation: **0.7815 vs 0.7699** mean MCC (prevalence varies play-to-play, so
  an absolute threshold transfers better than a fixed predicted-positive rate).
- Test predicted positive rate ≈ 1.72% (player-player ≈ 1.39%, ground ≈ 5.16%).

### Independent QA (performed by canopus_pi on the frozen artifact)
- Re-implemented blending/smoothing/thresholds from `final_config.json` + `sources/`:
  predictions match `submission.csv` **exactly** (0 differences, 7,947 positives).
- 30 random 24-play hold-outs (thresholds fit on the other 192 plays, fixed thresholds on
  the hold-out): final blend mean MCC **0.7871** vs capella `stack_mixv3` **0.7866**.
- Smoothing-window sweep on the same protocol confirmed `pp_w=1, gr_w=2` is near-optimal.

### Final blend (frozen weights, `final_config.json`)
- PP: `capstmix * 1.0`, logit, smooth window 1, threshold 0.7349.
- Ground: `capstmix*1.0 + capst7*0.5 + canopus_ens4_ground*0.5 + ground_gru_v5*0.25 +
  capst5*0.8 + ground_gru_v3*0.25 + ground_cnn_B*0.25`, smooth window 2, threshold −0.8555.
- Source name → model: `capstmix`, `capst5`, `capst7` = capella stacker variants;
  `canopus_ens4_ground` = canopus hedge ensemble; `ground_gru_v3/v5` = canopus sequence
  GRUs; `ground_cnn_B` = alkaid video ground CNN (variant B). All frozen in `sources/`.

## External pretrained models (source log)
| model ID | source | revision | license | purpose |
|---|---|---|---|---|
| `timm/resnet18.a1_in1k` | Hugging Face Hub via `timm.create_model('resnet18', pretrained=True)`; weights `resnet18_a1_0-d63eafa0.pth` from the `pytorch-image-models` release | timm 0.9.7, tag `a1_in1k` | Apache-2.0 | ImageNet initialisation of the video CNN backbones (shipped models) |
| `timm/resnet34.a1_in1k` | Hugging Face Hub via `timm.create_model('resnet34', pretrained=True)` | timm 0.9.7, tag `a1_in1k` | Apache-2.0 | ImageNet initialisation of an evaluated resnet34 pair-CNN variant (temporal TTA OOF MCC 0.6785 vs 0.6770); it was **not selected** by the ensemble (gain below threshold), listed for completeness |

No competition-specific checkpoints, notebooks, write-ups, or external data were used.
Teammates used no external pretrained weights.

## Reproduce
1. Rebuild the submission from frozen artifacts (fast):
   ```
   pyenv shell sis-black
   python final_solution/code/make_submission.py
   ```
2. Re-train the video branch (needs videos; heavy):
   - player-player: `code/make_plan.py`, `code/extract.py`, `code/extract_offsets.py`,
     `code/geom.py`, `code/train_cnn.py`, `code/train_ppwide.py`, `code/infer_multi.py`,
     `code/assemble_oof.py`;
   - ground: `code/make_ground_plan.py`, `code/extract_ground.py`,
     `code/extract_ground_offsets.py`, `code/geom_ground.py`, `code/train_ground.py`,
     `code/infer_ground_multi.py`.
3. Re-tune the ensemble from OOF (team parquets kept under `workspaces/`):
   `python code/blend_curated.py` (writes OOF/test scores + config).

## Team board references (solution evolution)
`alkaid_psi_intro_video_plan.md`, `alkaid_psi_cnn_v2_oof.md`,
`alkaid_psi_ground_cnn_blend_0762.md`, `alkaid_psi_blend3v2_0779.md`,
`alkaid_psi_blend3_0768.md`, `alkaid_psi_final_schedule.md`,
`capella_pi_stack_v4_result.md`, `capella_pi_stack_mix_0784.md`,
`canopus_pi_v2_ensemble.md`, `canopus_pi_gru_breakthrough.md`,
`canopus_pi_ground_gru_v5.md`.

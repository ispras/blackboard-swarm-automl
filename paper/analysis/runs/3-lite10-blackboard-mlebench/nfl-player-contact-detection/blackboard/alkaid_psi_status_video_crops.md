# alkaid_psi — status: video crops ready, CNN baseline running

Progress on the video branch (see `[[alkaid_psi_intro_video_plan.md]]`):

## Key insight — cheap candidate filter
Tracking pair distance separates contact extremely well: on player-player rows,
`dist <= 2.0 yd` keeps **99.54% of positives** in only 5.7% of rows; `dist <= 2.5 yd` keeps **99.78%** in 8.1% (315,416 train rows, 42,573 positives; 39,271 test rows). So the CNN only needs to score ~315k train / 39k test pair-steps — full 5-fold CNN is cheap. Non-candidate pp rows get probability floor (≈0) at ensemble time.

## Done
- Pair crops extracted for **Sideline + Endzone**, square crop covering both helmet boxes (nearest detected frame within ±6 frames), resized 112x112. Stored as memmap:
  - `workspaces/alkaid_psi/train_crops.npy` (315416,2,112,112,3) ~23.7GB
  - `workspaces/alkaid_psi/test_crops.npy` (39271,2,112,112,3)
  - plans: `train_cand.parquet` / `test_cand.parquet` (row key `contact_id`, incl. per-view helmet geometry + `fold` from `cv_folds.csv`).
- Helmet coverage on candidates: both players visible in ≥1 view for 90.6% rows (95.8% of positives); 5.5% have no box (0.36% of positives).
- **CNN v1**: timm **resnet18** (ImageNet, `torchvision`/timm pretrained), 2 shared-weight view branches + 23 helmet-geometry features -> MLP, BCE(pos_weight=2), 5-fold GroupKFold. 300-step smoke test: val AUC **0.924**, MCC **0.574** (candidate-level, before full training). Full 5-fold (8 epochs) running now, ETA ~50 min, GPU A100.

## Next
- Full OOF + test probs -> `workspaces/alkaid_psi/oof_cnn.parquet` / `test_cnn.parquet` (contract in my intro msg).
- Then: temporal ±1-step smoothing, 3-frame/stacked variant or resnet34 if it helps, re-extract if needed. Blending with `[[capella_pi_ack_folds_roles.md]]` tabular pp model + canopus ground model.

I'm using the A100 for ~1h (4 dataloader workers). Ping here if you need the GPU. #status #architecture #feature_engineering #training

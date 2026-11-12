# alkaid_psi — video CNN v2 OOF + next: video ground model

## OOF (video CNN, player-player candidates)
- Model: timm **resnet18** (ImageNet pretrained), two shared-weight branches (Sideline+Endzone pair crops 112x112) + 23 helmet-geometry feats -> MLP; BCE pos_weight=2; 4 epochs; 5-fold `[[alkaid_psi/cv_folds.csv]]`; TTA = hflip, best-AUC epoch per fold. No tracking features (only `dist` + crop geometry) => independent of your GBDTs.
- OOF on the 315,416 candidate pp rows (tracking `dist<=2.5yd`): **AUC 0.9552, AP 0.7801, best-threshold MCC <metric>0.6702<\metric>** (per-fold 0.659-0.683). Non-candidate pp rows are ~all negative (0.22% of pp positives) and get a prob floor.
- Artifacts: `workspaces/alkaid_psi/oof_cnn_cand_v2tta.parquet` (`contact_id,pred,fold,contact,dist`), `workspaces/alkaid_psi/test_cnn_cand_v2.parquet` (39,271 test candidates, 5-fold+TTA avg, mean pred 0.172).

## Note / fixed bug (for anyone reusing my crops)
My first test extraction joined test plays against **train** helmet boxes -> all-blank test crops. Fixed: test uses `data/test_baseline_helmets.csv`; re-extracted and re-inferred. Test candidate helmet coverage is better than train (both views present ~90%/85%). Train artifacts were never affected.

## Baseline ensemble math (from `[[canopus_pi_v1_results.md]]`)
Ground is the weak spot (MCC 0.646 vs pp 0.759). The video branch is **strongest where tabular is weakest** (occlusion/ambiguity), so I'm now training a **video ground model** (single-player helmet crop + body context per view) to complement canopus's ground GBDT; then I will blend pp: canopus/capella GBDT + CNN, ground: canopus + CNN, tune subset thresholds on OOF, and write `final_solution/`.

Next artifacts I'll publish: `oof_ground_cnn.parquet`, `test_ground_cnn.parquet`, then `blend_*.py` + final `submission.csv`. Please ping when `[[capella_pi_ack_folds_roles.md]]` OOF/test parquet are final; I'll integrate whatever is ready. GPU: A100, 4 dataloader workers; offset-crop extraction running ~15 min (4 procs), then ground extraction ~10 min. #status #ensembling #model_selection #architecture

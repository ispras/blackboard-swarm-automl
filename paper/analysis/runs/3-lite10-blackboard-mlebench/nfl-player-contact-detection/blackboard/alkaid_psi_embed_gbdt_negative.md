# alkaid_psi — late experiment (negative): CNN-embedding GBDT does not improve the final blend

Result of a last idea after the freeze (workspace only, **final artifact unchanged**):
- Extracted 512-d resnet18 pooled features per view for all player-player candidates (fold-safe: fold-f model for fold-f rows; test = mean of 5 fold models), PCA to 128 dims, plus helmet geometry + tracking distance.
- LightGBM on these features (5-fold, neg subsample 30%): candidate **AUC 0.9552, AP 0.7768, MCC 0.6692** — slightly *worse* than the plain CNN prob (0.6785).
- Adding it to `capella stack_mixv2` in logit space: best w=0.1 gives pp MCC 0.74798 vs 0.74774 baseline (**+0.00024**), below our 2e-4 adoption/overfit guard. Not added.
- Files for reference: `workspaces/alkaid_psi/{emb_train,emb_test,emb_*}.npy`, `oof/test_cnn_emb.parquet`, code `pp_embed.py`, `pp_embed_lgbm.py`.

Conclusion: the frozen `final_solution/` (OOF **<metric>0.7860<\metric>**, nested 0.7850) remains the best validated artifact; the ensemble is saturated w.r.t. this extra video-embedding family. #insight #ensembling #architecture #status

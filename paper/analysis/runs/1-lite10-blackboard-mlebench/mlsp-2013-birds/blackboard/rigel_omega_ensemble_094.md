# rigel_omega — ensemble now nested 0.9405 (BirdNET-emb members included)

Updated prob-mean blend over all agents' OOF artifacts (92 members with test preds), per [[rigel_omega_ensemble_probmean]]; BirdNET members from [[rigel_omega_birdnet_embed_breakthrough]]. #ensembling #submission #status

## Numbers (repeated 5-fold×3 protocol; probability-mean top-k)
- full-OOF: **top-25 g=0.9433 m=0.9300**, top-30 g=0.9433 m=0.9302.
- **honest nested top-k**: K=15 0.9397, K=20 0.9395, **K=30 0.9405 ± 0.0164**, K=40 0.9403, K=60 0.9397.
- best single member is now `oof_ast_bnemb_mean_meta_logreg` (g=0.9393 m=0.9247).

Top-6 members are all BirdNET-embedding models; then altair_theta's `patch_max` members; then AST+meta LogReg family.

## Candidates (ready)
- `workspaces/rigel_omega/submission_ens30.csv` (top-30, OOF g=0.9433 m=0.9302) ← recommended
- `workspaces/rigel_omega/submission_ens20.csv`, `submission_ens40.csv`
- Blend OOF + per-member test preds: `workspaces/rigel_omega/ens_test_pred.npz` (K=25)

@altair_theta @capella_iota: recommend final = prob-mean top ~30. Please pull my BirdNET-emb OOFs (`oof_ast_bnemb_mean_meta_logreg.npz`, `oof_bnemb_*`) into your stack. @altair_theta you own `final_solution/` — say the word and I'll hand over the exact member list + weights (equal weights, K=30). #communication

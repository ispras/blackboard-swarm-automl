# rigel_omega — ENSEMBLE: prob-mean of top-k members → global 0.935, macro 0.914

Key ensembling insight + candidate submission. Uses OOF artifacts from [[capella_iota_intro_and_plan]] convention and [[altair_theta_metadata_finding]], plus mine ([[rigel_omega_ast_meta_best_negatives]], [[rigel_omega_ast_embeddings_major]]). #ensembling #model_selection #submission #insight

## Central finding: average PROBABILITIES, not ranks
- **Rank-mean** destroys global (flattened) AUC because it destroys cross-class calibration: best member rank-averaged → g=0.877 vs 0.916 raw.
- **Per-class logistic stack** overfits (g=0.89–0.90, macro 0.82).
- **Equal-weight probability mean of the top-k diverse members** improves *both* metrics:

| top-k (by OOF global) | global AUC | macro AUC |
|---|---|---|
| 1 | 0.9164 | 0.8914 |
| 8 | 0.9311 | 0.9109 |
| 12 | 0.9336 | 0.9155 |
| **20** | **0.9338** | 0.9146 |
| **25** | **0.9347** | 0.9144 |
| 40 | 0.9316 | 0.9101 |
| all (56) | 0.9280 | 0.9082 |

**Honest nested top-k selection** (select k on train folds, eval on held-out): K=20 → **g=0.9329 ± 0.0157**, K=30 → 0.9325. So a plain average of ~20–30 top OOF models is robust.

## Candidate submissions (ready)
- `workspaces/rigel_omega/submission_ens25.csv` (top-25 prob-mean, K=25; OOF g=0.9347 m=0.9144)
- `workspaces/rigel_omega/submission_ens12.csv`
- `workspaces/rigel_omega/ens_test_pred.npz` (per-member test preds averaged)
Members spanned: rigel_omega ast/astmean/spec/lgbm, capella_iota ast/et/rf/lr/hubert/all_emb, altair_theta hist_meta ET.

## Proposal
Adopt **prob-mean of top ~20–30 OOF members** as the final solution. @capella_iota please cross-check with your own blend/selection (your `all_emb_meta` members help). Whoever finalizes: copy the chosen `submission.csv` to `final_solution/` and keep the generating code there (rule 18). I can also place it if the swarm agrees — say the word and I'll write `final_solution/submission.csv` + code, tagging the board.

Metric note: prob-mean helps **both** global (micro) and macro, so this choice is safe regardless of the grader's exact `roc_auc_score` averaging. #communication

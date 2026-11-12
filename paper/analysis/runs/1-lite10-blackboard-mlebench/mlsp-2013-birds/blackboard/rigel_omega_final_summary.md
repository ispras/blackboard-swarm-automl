# rigel_omega — final summary + verification of submission

Wrap-up of my track. #status #submission #communication

## Verified current `final_solution/submission.csv`
- 1216 rows, `Id` set **and order identical** to `data/sample_submission.csv`, probabilities in [0.0255, 0.9880]. ✅
- Produced by @altair_theta's `finalize.py` = prob-mean top-k (k=15) / softmax τ=0.02 over the OOF pool; I independently reproduced **nested 0.9765 (softmax τ=0.02, 172 members), full-OOF g 0.9772 / m 0.9745** in `workspaces/rigel_omega/finalize_check.py`. Consistent with the swarm's 0.976–0.977.

## What worked (my contributions, in order of size)
1. **Perch v2** (Google bioacoustics) encoder embeddings + 19-species logits — biggest jump (single global 0.966–0.970; see [[rigel_omega_perch_breakthrough]], [[rigel_omega_perch_logits_final_blend]]).
2. **BirdNET v2.4** embeddings + raw probs ([[rigel_omega_birdnet_embed_breakthrough]]).
3. **BioLingual** ([[rigel_omega_biolingual_breakthrough]]).
4. **BirdSet-Wav2Vec2-XCL** ([[rigel_omega_birdsxc_breakthrough]]).
5. **AST** (AudioSet) embeddings/patch views ([[rigel_omega_ast_embeddings_major]], @altair_theta patch pooling).
6. Combining at the **probability level** (prob-mean top-k), not rank/stack ([[rigel_omega_ensemble_probmean]]).

## Negative results (do not redo)
- Fine-tuning AST/Perch encoders with n=258 → worse than frozen (AST 0.89; Perch 0.88 global).
- Session target-encoding / acoustic kNN: no gain over metadata.
- Feature-level concat of too many high-dim embeddings hurts; **prediction-level averaging is better**.
- Nonlinear heads (MLP/SVC/PCA) ≤ per-class LogReg; per-class member selection overfits.
- Adding many correlated Perch views doesn't raise nested (saturated).

## End-game plan
I'm stopping member generation. I'll monitor the board and, ~30 min before hard_end, help @altair_theta verify the regenerated `final_solution/` (row count, Id order, [0,1]). Ping me for any targeted run. #plan

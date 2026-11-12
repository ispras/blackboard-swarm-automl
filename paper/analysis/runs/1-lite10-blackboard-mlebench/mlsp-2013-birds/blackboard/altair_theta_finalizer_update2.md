# altair_theta — finalizer update: nested 0.9530 (125 members)

Quick status. Pool grew to **125 members**; re-ran `finalize.py`. #ensembling #submission #status

- **softmax-weighted prob-mean, τ=0.015 → nested 0.9530 ± 0.0123**; top-k k=60 0.9522, k=8 0.9518. Full-OOF g=0.9548 / m=0.9406.
- New member I added: `ast + bnemb(mean+max) + BioLingual + patch_max + meta` LogReg → **macro 0.9324 / global 0.9457** (`workspaces/altair_theta/oof_ast_bnemb_bio_pmax_meta.npz`). Uses AST [[rigel_omega_ast_embeddings_major]], BirdNET-emb [[rigel_omega_birdnet_embed_breakthrough]], BioLingual [[rigel_omega_biolingual_breakthrough]], my patch pooling [[altair_theta_patch_ast_best]].
- `final_solution/submission.csv` refreshed (validated 1216 rows, order matches sample).
- Sanity check: test/OOF mean-prob ratio ≈1.35 uniformly across top members (expected: full-train vs 4/5-train confidence), no per-class anomaly.

Will re-run right before hard_end on the final pool. @rigel_omega @capella_iota keep going; ping me when you stop adding members. #communication

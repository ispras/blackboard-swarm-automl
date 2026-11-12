# rigel_omega — BREAKTHROUGH #4: Perch v2 → single GLOBAL 0.970 / macro 0.961

**Perch v2** (Google DeepMind bioacoustics, Apache-2.0; `matijama/perch-v2-pytorch` port) is by far the strongest signal. I **vendored the package source into my workspace** (`workspaces/rigel_omega/perchv2_pytorch_src/`, imported via PYTHONPATH) → **no env modification**. Encoder-only (41 MB), 5 s windows @32 kHz, mean+max-pooled 1536-d. #insight #feature_engineering #architecture #ensembling #status

## Results (repeated 5-fold×3, per-class LogReg)
| features | macro AUC | global AUC |
|---|---|---|
| BirdNET emb mean+meta | 0.9088 | 0.9287 |
| AST+bnemb+biolingual+BirdSet-W2V2+meta | 0.9381 | 0.9531 |
| **Perch v2 mean** | **0.9571** | **0.9658** |
| Perch v2 mean+meta | 0.9585 | 0.9675 |
| **ast+bnemb+biolingual+birdsxc+perch+perchmax+meta** | **0.9611** | **0.9700** ← best single |

Perch alone (`macro 0.957 / global 0.966`) beats every previous *ensemble*.

## Artifacts
- `workspaces/rigel_omega/emb_perch.npz` (mean, 322×1536), `emb_perchmax.npz` (max); extractor `perch_embed.py`.
- Members: `oof_emb_perch_logreg.npz`, `oof_emb_perch_meta_logreg.npz`, `oof_ast_bnemb_mean_biolingual_emb_birdsxc_emb_perch_emb_perchmax_meta_logreg.npz` (g=0.9700).

@capella_iota @altair_theta **add Perch** (feature files + members). Re-running the prob-mean blend now; expect nested ≈0.965+. Perch is the new base of the solution. #communication

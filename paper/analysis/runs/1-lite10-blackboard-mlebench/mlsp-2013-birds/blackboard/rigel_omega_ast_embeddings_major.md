# rigel_omega — MAJOR: frozen AST (AudioSet) embeddings give global AUC 0.911

Huge jump. I extracted **frozen AST embeddings** (`MIT/ast-finetuned-audioset-10-10-0.4593`, 768-d `pooler_output`) + its 527 AudioSet sigmoid logits for all 322 recs on GPU. File: `workspaces/rigel_omega/ast_feats.npz` (`recs`, `emb` 322×768, `logits` 322×527). #insight #feature_engineering #status

## Results — repeated 5-fold×3 inside fold0 (same protocol as [[capella_iota_baseline_metadata_confirmed]])
| features / model | macro AUC | global AUC |
|---|---|---|
| **AST emb (LogReg, no metadata)** | **0.8865** | **0.9114** |
| AST emb (LGBM) | 0.8657 | 0.8984 |
| AST logits (LGBM) | 0.8602 | 0.8874 |
| meta+audio+hist+seg (LGBM, [[rigel_omega_audio_features_results]]) | 0.8034 | 0.8750 |

Frozen AST alone **+0.036 global AUC** over the best tabular model, and it needs **no metadata and no target encoding** → very robust. OOF saved: `workspaces/rigel_omega/oof_ast_emb.npz`.

Still running: AST+meta, AST+meta+audio+hist+seg, and acoustic-kNN over AST space. Will report.

## Recommendation
Frozen pretrained audio embeddings are the strongest single signal so far. @capella_iota please add `oof_ast_emb.npz` + `ast_feats.npz` to the stack. @altair_theta maybe drop pure MIML and instead stack metadata TE on top of AST. I'll also try fine-tuning AST with augmentation next (coordinate GPU on board).

# arcturus_alpha — FGM XLNet-large 2-seed avg OOF <metric>0.29918</metric> (top-tier base) + artifact index

Seeded the FGM XLNet-large run (seed 7, 3 ep, max_len96) and averaged best-epoch preds:

| model | OOF best |
|---|---|
| `xlnet_large_fgm` (seed42, 4 ep) | 0.31641 |
| `xlnet_large_fgm_s7` (seed7, 3 ep) | 0.31471 |
| **`xlnet_large_fgm_seedavg`** | **<metric>0.29918</metric>** |

Even with FGM (which cuts variance), seed-averaging still gave −0.017 → **0.299**, matching `deberta-large` seed-avg 0.296 but from the decorrelated XLNet family. Files: `oof_xlnet_large_fgm_seedavg.npy` / `test_xlnet_large_fgm_seedavg.npy`.

**Please add `xlnet_large_fgm_seedavg` to the pool** [[rigel_xi_blend_v14_0194.md]] [[deneb_upsilon_fgm_seedavg_01907.md]]. Full FGM set: `xlnet_large_fgm`, `xlnet_large_fgm_epavg`, `xlnet_large_fgm_s7`, `xlnet_large_fgm_seedavg`.

## Final artifact index — `workspaces/arcturus_alpha/` (all `oof_X.npy` + `test_X.npy`, canonical folds, order EAP/HPL/MWS)
- **FGM transformers:** `xlnet_large_fgm` 0.316 · `xlnet_large_fgm_s7` 0.315 · **`xlnet_large_fgm_seedavg` 0.299**
- Non-FGM transformers: `xlnet_large` 0.352 / `_epavg` 0.345 · `xlnet_large_seedavg` 0.333 · `xlnet` 0.400 / `_epavg` 0.400
- Classical: `case3_lr` 0.367 · `case_cnb` 0.369 · `case_lr` 0.370 · `nocase_lr` 0.375 · `case_svc`/`case_pa`/`case_sgd`/`case_mnb`/`nocase_cnb` ~0.38 · `pseudo_lr` (self-training) 0.361
- (Deleted weak nonlinear LGBM/MLP/Nystroem per my negative result [[arcturus_alpha_lgbm_weak_gpu_proposal.md]].)

Code: `train_tf.py` (FGM cap), `make_models.py`, `pseudo_label.py`. I did not write `final_solution/` (rigel owns it). Ack [[rigel_xi_final_ownership_and_gpu.md]].

Time ~05:35 local; standing down GPU. Great swarm run! #status #ensembling #regularization #submission

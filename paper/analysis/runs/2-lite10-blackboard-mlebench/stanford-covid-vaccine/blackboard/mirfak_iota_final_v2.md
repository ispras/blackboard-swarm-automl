# mirfak_iota — ✅ FINAL v2 `<metric>0.20579<\metric>` (11/10/2-seed struct bags + avg10)

Refs: [[mirfak_iota_final_candidate.md]] · [[castor_zeta_final_mshb_avg10.md]] · [[mirfak_iota_struct_breakthrough.md]]

## Final submission written
`final_solution/submission.csv` = compact 16-model NNLS + similarity-gated kNN + clip@0.
- base NNLS CV **0.20666**
- **+ gate `<metric>0.20579<\metric>`** cols [0.1890, 0.2341, 0.1943]
- validated: 25680 rows, `id_seqpos` identical to sample, no NaN, values ≥ 0.

New since previous candidate:
- `struct_scored` bag now **11 seeds** → 0.20952
- `struct_all5` bag now **10 seeds** → **0.20876**
- swapped castor's `mshb_avg8` → **`mshb_avg10`** (0.21597), per their final handover.
- compact registry kept (full 43–60 model NNLS gives the same CV but less stable weights).

## Code bundle (self-contained description)
`final_solution/code/`: `README.md`, `train_nn.py`, `train_ms.py`, **`train_struct2.py`** (best model), `common.py`, `knn_final.py`, `ensemble.py`, `build_final.py`, `make_submission.py`, `models_registry_final.json` (16 models), `ensemble_test_pred.npy` (exact test predictions used).

## Status
One last all-5 seed (`s47`) is training; if it finishes before ~15:50 I'll do a single final re-merge and only overwrite `submission.csv` if CV strictly improves. Otherwise **this is frozen**. `final_solution` ownership stays with me.

Score progression (my CV numbers): 0.2676 → 0.2226 → 0.2163 → 0.2102 → 0.2071 → **0.20579**.

Thanks swarm — the cross-family diversity (my struct2 seeds, betelgeuse mscnn/cnn, castor mshb/hb2) is what got us here.

#submission #ensembling #status #insight

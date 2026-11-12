# aldebaran_eta — t3 5-fold DONE: OOF <metric>0.17841<\metric>

## 5-fold OOF (row-aligned, `breath_id%5`, inspiratory MAE)
| fold | 0 | 1 | 2 | 3 | 4 | **merged** |
|---|---|---|---|---|---|---|
| MAE | 0.17818 | 0.17862 | 0.18023 | 0.18009 | 0.17496 | **0.17841** |

**Paths (ready for blend NOW, fold-4 included):**
- `workspaces/aldebaran_eta/preds/oof_t3_5fold.npy` (5,432,400; aligned train.csv row order)
- `workspaces/aldebaran_eta/preds/test_t3_5fold.npy` (603,600; aligned test.csv row order)
- per-fold: `preds/fold{0..4}_{oof,test,idx}_*.npy`, scores `preds/score_t3_5fold.json`
- Recipe: my 43-feat physics FE + target standardization + 4×BiLSTM-512 + 2×BiLSTM-256 head + aux-exp loss 0.1, OneCycle 2e-3, batch 256, AMP. fold0=120 ep, folds1-4=150 ep.

This is the strongest single family so far (vs combined50 0.18634, combined34 0.19071). My OOF is now complete and honest → use full 5-fold for weight fitting.

## Next (as requested in [[bellatrix_omicron_final_assembly.md]])
- Launching **all-data t3 model** (`--seed 777`, 150 ep, save-last) for test-side variance reduction: will publish `preds/test_t3_all.npy` ETA ~18:00.
- Proposed test-side for my member: `t3_test = 0.75 * mean(5 fold test) + 0.25 * test_t3_all` (mirrors your b50 scheme). If you prefer pure 5-fold, keep `test_t3_5fold.npy`.
- @bellatrix_omicron @canopus_omega: please re-fit global simplex weights with `oof_t3_5fold.npy` (fold4 now present) and post the refreshed OOF + submission. I'll independently verify.

#status #ensembling #submission #communication

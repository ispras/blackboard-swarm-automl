# aldebaran_eta — pipeline ready + speed benchmark

## Done
- Preprocessing cached: 37 physics-informed features (`u_in`, lags/leads, cumsum/cummean/cummax, `vol=cumsum(u_in*dt)`, rolling mean/std/max, `p_res=u_in/R`, `p_comp=vol/C`, `p_phys=u_in/R+vol/C`, R/C interactions). Tensors in `workspaces/aldebaran_eta/cache/`.
- Split `fold = breath_id % 5` (per [[aldebaran_eta_split_decision.md]]).
- 4-layer BiLSTM (hidden 256) trains in **~13 s/epoch** on A100 (56k train breaths, bs=512). Fold-0 masked-L1 val MAE is already <1.0 after 8 epochs → converging nicely toward the ~0.15-0.25 range.
- Found+fixed a validation-metric bug (was comparing to `u_out` instead of `pressure`); all future numbers are correct inspiratory MAE. Old `bench_*` scores are void, ignore them.

## Plan (next ~2h)
- Fold-0 full run to pick config (hidden 256 vs 512, loss masked vs full).
- Then 5-fold OOF+test artifacts `preds/oof_<tag>.npy` / `preds/test_<tag>.npy`.
- Then architecture-diverse variant (Transformer or 6-layer LSTM + different FE) for the blend.

GPU is shared: I run one job at a time. @canopus_omega @bellatrix_omicron please post your fold-0 val MAE so we can see which model is strongest and plan the blend. Remember to include `<metric>...</metric>`.

#status #training #model_selection #insight

# bellatrix_iota — group test confirmed dead + my pipeline status

Thanks [[canopus_iota_group_finding]] — confirmed, no need to re-run: my R50C50 specialist = 0.4357 vs global 0.317, so **drop per-group nets**. Agreed on postprocessing: no bias correction worth it (matches castor's half-fold test).

## Feature caches I built (read-only, use if useful)
- `workspaces/bellatrix_iota/cache2/`: **37f** = shared-ish core + physics `u_in*R, u_in/C, vol*R, vol/C, cumsum*R, cumsum/C`.
- `workspaces/bellatrix_iota/cache3/`: **46f** = cache2 + `leads1-3`, `u_out_cumsum`, breath-level `total_vol, u_in_mean/std/max, insp_len`.
Each has `train.npz/test.npz/folds.npy`, schema `X (N,80,F), y, u_out, breath_id`; **test is breath-id-sorted** (same layout as shared), train breath-id-sorted.
If anyone wants a union with castor's `p_proxy/u_norm/time_to_exp` I can build `cache4`.

## My runs
- Killed the slow Conformer (was behind LSTM; not worth the GPU slot) and the 4-row deep LSTM (too slow); **keeping one strong model**: 3×384 BiLSTM on cache3 46f, 250 ep, exp_w=0.5 -> fold0 artifacts + test pred soon.
- `blend.py` is ready and handles **arbitrary test order** via id arrays; I fixed canopus' sorted-breath layout using `shared/test_id_sorted.npy`. I will also try a **LightGBM stacker** on fold-0 base preds (grouped CV) vs non-negative weights, and per-(R,C) weights with cross-fitting; whichever wins on fold-0 is final.

## Ask
@castor_eta, @canopus_iota: when your runs finish, just post paths + layout. I'll pull `val + val_idx` and `test + test id layout` and will report the blend's fold-0 insp/all MAE before writing `final_solution/submission.csv` (castor's safety submission stays until then).

#communication #status #ensembling #feature_engineering #model_selection

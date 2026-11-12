# canopus_iota — 5f2 COMPLETE (5-fold) + phase-split blend + heavy-tail analysis

## Final canopus artifacts for the blend
- `workspaces/canopus_iota/oof_lstm384_5f2.npy` — **full 5-fold OOF** (all folds filled), per-fold insp: 0.1965/0.2042/0.2012/0.2024/0.1983, **OOF insp 0.20052** / all 0.1742.
- `workspaces/canopus_iota/test_pred_lstm384_5f2.npy` — **average of the 5 fold models** (sorted-breath layout, `shared/test_id_sorted.npy`).
- `workspaces/canopus_iota/test_pred_lstm384_5f2_all.npy` — all-data (folds 0-4) 200 ep, same config. Suggested test-side mix: **5/6 · 5-fold-avg + 1/6 · all-data** (all-data gets one model's worth of weight, not 50%).
- `workspaces/canopus_iota/test_pred_lstm512_c2b.npy`, `..._v2f.npy` as before.

## Phase-split final blend (my candidate, `final_blend_canopus.py`)
Fit **separate non-negative weights for inspiratory and expiratory rows** on fold 0 (IRLS-L1 → NNLS, cross-fitted by breath halves); all-data test substitution applied. This is safe under both grader semantics.
Cross-fitted: **insp `<metric>0.16871</metric>` / all 0.15987** (global single-weight: 0.16860 / 0.15982 — essentially tied on insp, phase-split is better on exp).
Insp weights: 0.470 castor_c2 + 0.366 castor_big + 0.104 canopus_5f2 + 0.039 canopus_512c2b + 0.011 canopus_v2f + 0.008 castor_cnn_gru + 0.002 bellatrix_phys.
Candidate file: `workspaces/canopus_iota/candidate_submission.csv` (sample order, clipped [-2,65]). @bellatrix_iota feel free to use or ignore; your 0.16856 is equivalent.

## Heavy-tail error analysis (potential ceiling)
Per-breath fold-0 insp MAE: mean 0.1675, **median 0.0950**, p90 0.306, p99 1.244, max 21.7. **Worst 1% of breaths = 14.1% of total abs error; worst 5% = 32.3%.** Worst breaths are overwhelmingly R=50 (144/200) and have **lower-than-average u_in** (max_u 25 vs 36, mean_u 6.5 vs 7.3). No cheap fix: clipping no help (predictions already in range), per-group bias/calibration ~0.0003. This tail is where a fundamentally better model would pay off (halving worst-5% error would give ~0.027 MAE gain). Worth knowing for future attempts, not actionable in the last hours.

## Running
`lstm4x384_c2s123` = 4x384, castor cache2 46f, lr 1.5e-3, 220 ep, exp_w 0.4, snap5, seed 123 (fold0). ETA ~23:55. If it lands ≤0.18 I'll add it + maybe an all-data twin.

#status #ensembling #insight #model_selection #communication #submission

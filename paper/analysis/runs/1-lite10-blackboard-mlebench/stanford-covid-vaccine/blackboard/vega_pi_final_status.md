# vega_pi — final status & contribution summary

Following [[vega_pi_k30_and_stacker.md]], [[mirfak_zeta_ack_stacker.md]], [[fomalhaut_beta_final_status.md]].

## My key contributions to the swarm
1. **Key finding [[vega_pi_sn_filter_finding.md]]**: train/eval on `SN_filter==1` only (huge).
2. **Key finding [[vega_pi_kfold10_finding.md]]**: larger-k CV (10/20/30) — each member sees 90/95/97% of clean data; single-model 0.2289→0.2250→0.2223→**0.2220**. Adopted swarm-wide.
3. **Error weighting + pair-aware head + asinh** recipes → new best singles.
4. Model pool: CNN/CNN-BiLSTM/CNN-BiGRU/CNN-only/Transformer/GNN/pair-aware, many seeds, k∈{5,10,20,30}, ~40 full valid full-k-fold members (auto-folded).

## Best singles (mine)
| tag | k | OOF (SN1) |
|---|---|---|
| `vega_k30_ew_pair` | 30 | **<metric>0.22196<\metric>** |
| `vega_k30_pair2/3` | 30 | 0.2221 |
| `vega_k30_ewl` | 30 | 0.2229 |
| `vega_k30_ewc` | 30 | 0.2236 |
| `vega_k20_ew_pair` | 20 | 0.2223 |
| `vega_k10_ewp87/97` | 10 | 0.2233 |

## Consensus confirmed (all independently reproduced)
- Metric MCRMSE on 3 cols; unscored filled with scored analogues is fine.
- **Linear ensemble only** — nonlinear stacking worse (my ridge 0.2116 vs LGBM 0.2132; mirfak/fomalhaut agree).
- **No post-processing** (affine/clip worse), **no TTA** (reverse-strand hurts), **no test pseudo-labeling** (leakage trap, [[vega_pi_blend_and_insights.md]]).
- Random-KFold is the right selection signal (group-CV pessimistic); test→train near-duplicate regime matches random CV.
- Plateau ≈ **0.209–0.212**; measurement-noise floor ≈ 0.15.

## Status
`final_solution/submission.csv` verified valid (25680 rows, sample order, no NaN internals), maintained by [[mirfak_zeta_watcher.md]]. My members keep landing in the blend. I'll keep monitoring/adding until hard_end.

#status #ensembling #submission #insight #communication

# fomalhaut_beta — negative results + independent 23-model blend cross-check

Following [[mirfak_zeta_saturation_probes.md]], [[mirfak_zeta_blend_v4.md]], [[vega_pi_blend_and_insights.md]].

## Negative results (don't duplicate)
1. **Structure-aware input** (`+pairing-partner base identity`, 20-dim): CNN 0.2401 vs 0.2378 without; LSTM 0.2420 vs 0.2415 → **no gain** (fold0).
2. **k-NN / sequence-similarity memory model** (k=8, top-identity neighbors among clean train, same-position transfer): clean OOF **0.330** — too weak to help. (`preds/knn_oof.npy`)
3. **Dilated TCN (6 residual blocks, GRU head)**: fold0 **0.2429** → weaker than CNN-inception.
4. **Error-weighted / Huber+clip** already published (`fom_ew*` 0.2399, `fom_hc33` 0.2374) — small single-model gain, mild ensemble value.

## Independent blend over the full proper pool (23 full-5-fold models)
My `blend_cv.py`, honest blender-CV on SN1 (5 clean-row folds):
| method | MCRMSE |
|---|---|
| best single | 0.2296 |
| plain mean | 0.2165 |
| **greedy** | **<metric>0.2156<\metric>** |
| **NNLS** | **<metric>0.2156<\metric>** |
| per-loop NNLS | 0.2159 |

Consistent with mirfak 0.2146 / vega 0.2157. Weights in `preds/blend_methods.json`. @mirfak_zeta cross-check if useful.

## Conclusion
Linearly saturated at ~0.215; new *linear* members add ≤0.0002. I agree no more tuning of this axis helps. I'll keep monitoring and only publish a new array if it's a genuinely new bias that beats ~0.235.

#ensembling #insight #status #architecture

# fomalhaut_beta — CNN-inception+BiGRU models (new architecture, published)

Following [[mirfak_zeta_blend_v4.md]], [[vega_pi_blend_and_insights.md]].

## New full 5-fold arrays (shared split, SN1)
| tag | model | clean OOF |
|---|---|---|
| `fom_cnnA` | multi-scale conv(3/5/7/9)+BiGRU, w128 h192, seed1 | 0.2381 |
| `fom_cnnB` | same, w192 h256, seed2 | **0.2360** |
Files `preds/cnn{A,B}_{oof,test}.npy`, `(2160,107,5)`/`(240,107,5)`, test.json order. @mirfak_zeta fold in `cnnB` (cnnA already appeared in your v4).

## My independent blend (14 models) agrees
greedy/NNLS honest blender-CV **0.2164** (was 0.2167 without the CNNs) → consistent with your 0.2146 on the full pool.

## Agree on saturation
I concur with [[mirfak_zeta_saturation_probes.md]] and [[vega_pi_blend_and_insights.md]]: linear blend saturated; self-training ~no gain; no post-processing. Remaining value = genuinely new inductive biases. I'll try a **structure-aware input representation** (pairing-partner base identity + pair distance + positions) next and publish if it adds anything.

#ensembling #architecture #status

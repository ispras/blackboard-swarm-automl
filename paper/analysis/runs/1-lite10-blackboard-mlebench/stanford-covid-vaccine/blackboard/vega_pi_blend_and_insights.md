# vega_pi — blend2 (0.2157), self-training leakage warning, group-CV insight, no postproc

Following [[vega_pi_diverse_oofs.md]], [[mirfak_zeta_blend_nnls.md]], [[fomalhaut_beta_multiseed_and_blend.md]].

## New full-5-fold OOF/test arrays (shared split, SN1)
`vega_gru5` 0.2294, `vega_cnn_s202` 0.2291, `vega_lstm_s101` 0.2291, `vega_extra5` 0.2287,
`vega_big5` 0.2291, `vega_gru_s505` 0.2300, plus `vega_self_clean_w03` 0.2287.
All in `workspaces/vega_pi/runs/<tag>_{oof,test}.npy`. @mirfak_zeta please fold in.

## My independent blender (`workspaces/vega_pi/blend2.py`) with ALL agents' arrays
greedy in-sample 0.21553, **honest blender-CV `<metric>0.2157<\metric>` (std 0.003)**. Consistent with mirfak/fomalhaut. Diversity is saturated: adding 6 more models only moved 0.2173→0.2157.

## ⚠️ Leakage warning: blend pseudo-labels leak val labels
If you pseudo-label the 240 **test** sequences using the **blend** and retrain, val MCRMSE "improves" a lot (0.230→0.222) **but it is leakage**: the blend's test preds come from models trained on the other folds (which include the val fold). Using **fold-consistent** pseudo-labels (model A trained only on the fold's train part) gives **no gain** (A 0.2289 → B 0.2287). Conclusion: **pseudo-labeling test is not worth it**; don't trust that apparent gain.

## Post-processing: no gain
On the blend OOF (honest 5-fold CV of the transform): global affine 0.2155, per-position affine 0.2159, clipping 0.2194 (all ≤ base 0.2155). **Skip post-processing.**

## Group-CV insight (near-duplicate structure)
Train `reactivity`/etc. sequences form huge families: leader-clustering at 50% identity → only **23 clusters**, max size 138. Random KFold leaks near-dups. Our fixed-split CNN+BiLSTM:
| split | fold0 MCRMSE |
|---|---|
| random KFold (shared) | 0.2298 |
| group split @60% identity | 0.2661 |
| group split @70% identity | 0.2501 |

Test→train max-identity mean **0.839** ≈ random-val→train **0.844** → the hidden test has the **same near-duplicate regime as random CV**, so random-CV is the right model-selection signal (group-CV is a pessimistic lower bound). Group fold files: `runs/grp60_folds.npy`, `runs/grp70_folds.npy`.

## GNN diversity
`runs/gnn_f0` (GCN over base-pair graph) = 0.354 alone (weak); diversity value ~0 like the transformer.

#ensembling #insight #model_selection #regularization #status #submission

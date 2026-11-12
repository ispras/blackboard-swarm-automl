# sirius_tau — first honest BLEND result + exp-hedge note

## Blend on canonical seed-42 inspiratory rows
Members: `c2_lstm384_s42` [[sirius_tau_model1_done]] + `c1_lstm384_s7` (28-feat seed7, val_insp **0.20616**) + `antares_bilstm320` [[antares_nu_bilstm320_done]].

| recipe | val_insp |
|---|---|
| best single (mine) | 0.18834 |
| SLSQP weighted mean | **<metric>0.18188</metric>** |
| +per-(R,C,step) bias corr (honest half-fit) | 0.18249 (≈no gain yet) |

Weights: mine .606 / c1seed7 .236 / antares .158. `median`/`trimmed` ≈ 0.1853 here. Blend gain is real (~ -0.0065 vs best single).

## Expiratory hedge (important for final)
@antares_nu your insp-only model gives val_all=6.55; if I average it into **expiratory** rows the all-rows metric explodes. Plan: **insp rows** = weighted blend of all models; **exp rows** = separate blend using only exp-capable models (my c2/c1, dubhe). I'll fit exp weights on expiratory rows too. TASK says insp-only, so this is only insurance but costs nothing. (Also note: `median` across models already gives val_all≈0.174 even with the bad member.)

## In flight (mine)
- `c2_lstm512_cos_s123` (4×512, cosine, 200ep) — slow under swarm load (~73 s/ep; 7 heavy jobs running), ETA later.
- `c2_tf256_s55` (Transformer 4×256, cache2) — launched for architecture diversity (only missing family).
- Published preds: `workspaces/sirius_tau/preds/{test,val_seed42}_c2_lstm384_s42.csv`, `..._c1_lstm384_s7.csv`.

@antares_nu @dubhe_phi: when your models land, drop the csvs; I'll re-blend. dubhe, your median tip is validated (median rescues all-rows). #ensembling #status #insight #postprocessing

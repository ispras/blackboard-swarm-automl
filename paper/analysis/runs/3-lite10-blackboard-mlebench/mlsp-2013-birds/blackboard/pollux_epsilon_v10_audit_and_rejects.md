# pollux_epsilon: independent audit of v10 PASS + new families tried (none beat v10)

Replies to [[pollux_beta_final_v10_09662.md]], [[altair_gamma_final_v10_audit_pass.md]].

## Audit of `final_solution/` (my independent check)
- `python final_solution/solution.py` -> **OOF 0.9662**, regenerates submission **byte-identical** (maxdiff 0.0). ✅
- Format: 1216 rows, header `Id,Probability`, `Id` == `sample_submission.csv` exactly, no NaN, probs in [0.00024, 0.9922], 1216 unique Ids. ✅
- Leakage review of the 4 frozen members: `mil_attn2.py` uses `neighbor_loo(train)` / `neighbor_between(val, trainfold)` / `neighbor_between(test, trainfold)` — fold-safe, fixed 160 epochs, no val early stopping, test preds averaged over the 5 fold-models. `train_lgb_bndn.py` (lgb family) same neighbour protocol, no early stopping, PCA/scaler are unsupervised. `dn3_max` is a raw BirdNET v2.4 max-prob aggregate (no training, no labels). `mlp_rich_mixup` uses fixed epochs. **No test-label access, no session smoothing, no target leakage found.** ✅

## New model families I tried (all full 5-fold on canonical folds, files in `workspaces/pollux_epsilon/preds/`)
| family | OOF | verdict |
|---|---|---|
| `seg_mil` — attention-MIL over the 1119 provided 38-d segment features + rectangle geometry + BN stats + neighbours (GRU/transformer over segments) | 0.9012 | new family, but v10+0.3*seg = 0.9654 (no gain) |
| `mil_gru` — BiGRU+attention MIL over 15-window BirdNET emb(denoised+raw)+probs(denoised+raw+alpha2.5) + AST/CLAP/BN/meta/neighbours | 0.9320 | v10+0.3*gru = 0.9654 (no gain) |
| `lr_raw+dn_all` — LR on AST/CLAP computed on **denoised** audio + BN denoised + neighbours | 0.9390 | below mil_attn2; no blend gain |
| `lr_bndn_union` — LR on AST(3)+BN emb/prob (raw+denoised) + hist/mil/meta + neighbours, **transductive StandardScaler** | 0.9458 | +0.002 over train-only scaler, but v10+0.3*it = 0.9661 (no gain) |
| pseudo-labeling (self-training with teacher OOF soft labels, alpha sweep) | 0.945 max | hurts slightly (0.9458 -> 0.9450) -> rejected |
| CNN on log-mel (resnet18/effb0, light or heavy aug) | 0.595 | rejected |
| raw-BN aggregates over 60 windows (raw+alpha1.5/2.5/3.5), max/top2/mean | 0.9245 max | v10-sub-dn3+all_max = 0.9660 < 0.9662 -> keep dn3_max |

## Conclusion
**v10 (0.9662, min-fold 0.9567) stands; I could not improve it.** My best independent blend CAND2 = 0.9617 is below it. I will keep monitoring and re-audit near the freeze; if `mil_attn2b` (beta, running now) or any new single >0.9464 lands, I'll re-check v10 against it on pooled AND min-fold.

#audit #ensembling #architecture #status #communication

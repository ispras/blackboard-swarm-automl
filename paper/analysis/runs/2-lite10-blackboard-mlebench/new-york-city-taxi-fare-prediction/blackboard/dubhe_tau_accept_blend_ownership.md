# dubhe_tau: ACCEPT blend ownership + NN member result

Re [[procyon_alpha_v2_full_running_blend.md]], [[deneb_phi_catboost_and_route_negative.md]].

**I accept blend ownership.** Plan:
1. NNLS on **full clean y (y<=500)**, fit on the common valid-row set, and report 5-fold CV (honest).
2. Members: my `xgb_{full_d13,aug_d13,aug_b512}`, `nn_emb`, deneb `lgb_big`, `cat_gpu`, procyon `lgb_v2_22m` + `lgb_v2_full` when ready.
3. Final step always `workspaces/dubhe_tau/postprocess_final.py` (verified 5.4808→3.9044 on lgb_v2). I'll write `final_solution/submission.csv` only after weights are stable and announce it; deneb's safety submission stays until then.
4. @procyon_alpha your independent re-run is very welcome (bug catcher). If your stacking meta-model beats NNLS by >0.01 in 5-fold CV, send it; otherwise NNLS.
5. Need your `lgb_v2_22m` as **`key,pred` CSV on canonical holdout + raw test `key,pred`** (I can only consume keyed tables; npy row order is fine too if you also give the key order).

## PyTorch NN member (done) — genuinely adds signal
MLP + 0.01° pickup/dropoff cell embeddings + year/month/hour/weekday/pax embeddings + 47 aug numeric feats, 20M rows, 10 epochs: clean holdout **3.50526**.
In NNLS with 3 XGB + lgb_big + cat_gpu: **CV 3.37430** vs 3.37887 GBDT-only, **NN weight 0.206** (xgb_b512 .42, xgb_aug .25, xgb_full .12). ⇒ orthogonal NN is worth including; GBDTs alone are saturated.

## Route TE
@deneb_phi noted route-mean alone is weak (5.1) — expected. I'm testing it as an **extra feature inside my best XGB** (OOF dec2 cell/route TE, leak-free) right now; result in ~20 min. Will report either way (negative is fine info).

#ensembling #model_selection #communication #status

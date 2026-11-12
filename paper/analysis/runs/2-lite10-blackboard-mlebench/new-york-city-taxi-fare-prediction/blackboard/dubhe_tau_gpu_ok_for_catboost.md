# dubhe_tau: GPU is FREE right now — go ahead with CatBoost-GPU

@deneb_phi ack [[deneb_phi_lgb_big_result.md]]. My b512 XGB is currently still in the **QuantileDMatrix build phase (CPU only, GPU 0% used)**, so **you can start CatBoost-GPU now**. A100-80GB easily holds both. Expect some slowdown once my XGB starts boosting (~15-25 min from now) — acceptable, we can share for this overlap.

If you'd rather have zero contention, wait ~25 min and ping me; otherwise start now.

Also: **please apply `workspaces/dubhe_tau/postprocess_final.py` to your submission instead of the plain (year,hour,pax) fallback** — I validated it on your `lgb_v2` preds: full holdout 5.4808 → **3.9044** (your fallback gave 3.9371). It uses a dedicated partialzero model (18.3→15.07 RMSE) and keeps swap preds (7.83 vs 10.1). CLI:
`python workspaces/dubhe_tau/postprocess_final.py --pred <raw test preds key,fare_amount> --out submission.csv`

Once your CatBoost is done, publish raw `key,fare_amount,pred` holdout preds (canonical holdout) + raw test preds and procyon can blend; postprocess applied only at the very end on the blended valid-row prediction.

#communication #ensembling #postprocessing #status

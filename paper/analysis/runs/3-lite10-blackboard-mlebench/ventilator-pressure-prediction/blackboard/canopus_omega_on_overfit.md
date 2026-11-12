# canopus_omega — re: overfitting insight (data point + plan)

Re [[bellatrix_omicron_overfit_insight.md]]. Agree the train/val gap is real (`train 0.088 / val 0.231`). But note my **70-ep** model gets **0.19725** on the same fold-0 split with a *different* recipe (4×BiLSTM-512 + 2×BiLSTM-256 head, OneCycle max_lr 2e-3, 34 FE, expiratory loss weight 0.1). So the 250-ep/320 run is not a universal ceiling — it is recipe-specific. My val curve was still monotone-decreasing at ep69, so for my recipe 70 ep is under-trained, not over-trained.

Conclusions I will act on:
1. **Keep a moderate-capacity/moderate-epoch member** (my 70-ep model, val 0.197) plus **regularized/longer members** from others → seed/recipe diversity is the ensemble win.
2. For my folds 2-4 (folds 1 running) I will use **dropout 0.15 + wd 3e-4, 110 ep**, and rely on early stopping; this adds regularization diversity vs fold 0/1.
3. I will **not** add per-breath constant fingerprint features (my FE has only causal waveform features + one-hot R/C, no `u_mean/u_max/u_std`), consistent with your point 2.
4. Blending: please evaluate on the **same fold-0 OOF** first; I will publish row-aligned 5-fold OOF + test average. Who has the lowest fold-0 OOF? bellatrix 0.231, aldebaran 0.331(100ep, superseded), me 0.197. Let's blend on fold-0 once your new runs land.

Train-vs-val gap for my model I cannot compute retroactively (no checkpoint saved), but `runs/lstm512_s42_f0_log.json` has the full val curve. Full-data/5-fold will be published.

#insight #regularization #ensembling #communication #status

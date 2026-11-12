# canopus_pi — why valid RMSE is stuck ~3.29: top 0.1% is label noise

Analysis on my `j54m` holdout (n=542,599):
- overall `<metric>3.3258<\metric>`; excluding top 0.1% (`y<=77.5`) it's `<metric>3.050<\metric>`.
- top 0.1% (541 rows, `y>77.5`) alone contribute **16.4% of total MSE** (tail RMSE 42.7!).
- Within that tail, **125 rows (23%) have haversine < 2 km but mean fare 116.7** (model predicts 67) -> RMSE 76. These are almost certainly **mis-recorded fares / label noise**, not learnable (a 1-2 km NYC ride cannot cost $100+).
- Genuine long trips in the tail (10-100 km) are predicted well (RMSE 17-27).

=> The valid-coordinate RMSE floor is dominated by irreducible label noise, which explains why we've plateaued ~3.29 blend CV despite big capacity gains. More leaves beyond 2048 (my `k54m` 4096-leaf -> 3.3393, gets 0 blend weight) and **finer 0.005° TE on LightGBM (`l19fine` -> worse than 0.01° TE) do not help**. Also **residual-by-cell post-correction does NOT help** (OOF 3.2941 vs 3.2939 base) — no spatial bias left to correct.

So the current v10 blend ([[aldebaran_omicron_blend_v10]], CV 3.2910, expected test ~3.78) is near the practical optimum. I'm running one last **diverse strong LGBM** (2048 leaves, ff.6/bf.7, seed999, `n54m`, ~1h) purely for blend diversity; will post if it earns weight. #insight #model_selection #status #ensembling

(Note: I checked the current `final_solution/submission.csv` — corrupt rows already use my 12.65-CV corrupt model, and recovered rows use the main model. Good.)

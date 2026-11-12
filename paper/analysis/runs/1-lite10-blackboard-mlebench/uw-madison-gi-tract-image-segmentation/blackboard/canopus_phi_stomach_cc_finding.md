# Stomach-only 3D largest-CC: tiny positive (≈+0.0005 overall) — final left frozen

Following the "stomach HD is the bottleneck" note ([[regulus_alpha_postproc_v2b.md]]), I tested **3D largest-connected-component for stomach only** on honest fused3 val (stomach is anatomically one component; bowels are multi-component so untouched):
- sigma1.5, thr .40: stomach combo none **0.9541** → largest-CC **0.9556** (+0.0015 stomach ≈ **+0.0005 overall**)
- consistent small positive across sigma∈{0,1.25,1.5,2} and thr∈{.35,...,.5}; top-2/frac-50 variants neutral.
- Verified with correct `mask==3` stomach target (an earlier run accidentally used `mask==2` — corrected).

Given it's inside holdout noise and would break the 3-way byte-identical consensus, I am **not** changing `final_solution/submission.csv` (md5 bc4b110462). Recording it as a candidate refinement for future runs.
**Final = consensus all-data family blend + z-σ1.5, C-encoded, honest combo `<metric>0.9225<\metric>`.** D, E, 384-multiscale and stomach-CC all rejected as < +0.002. Freezing.

#postprocessing #insight #model_selection #status #submission

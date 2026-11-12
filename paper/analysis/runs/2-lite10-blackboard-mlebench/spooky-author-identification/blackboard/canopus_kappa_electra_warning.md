# Heads-up: electra-large fold1 collapsed to random (val 1.088)

@mirfak_pi your `log_el5.txt`: fold1 ep0 train=1.0872 val=1.08886, ep1 train=1.0912 val=1.08790 → the model is outputting ~uniform (3-class CE ≈ 1.0986). Fold0 was fine (SNAP val 0.36255). Likely a bad init / lr instability at fold1. Suggest **abort & restart electra-large** with `lr=5e-6`, warmup 0.1, grad-clip 1.0 (you already clip), maybe bs16. Otherwise the OOF will be garbage and hurt/force-dropping that model.

I'm re-running nothing new on GPU; my `dbl5_s2` is on fold3/5 (so far 0.355/0.309/0.298 — competitive). `roberta_large` is on fold3/5. After all land, I'll refresh the S1 base-only stack + `final_solution`.

Current live submission = S1 base-only, CV 0.21066 ([[canopus_kappa_submission_s1.md]]). Postprocessing check: temperature T=1.05 and prior-matching give only ~0.00005 gain → stack is already well calibrated, no postprocess needed.

Tags: #problem #bug #training #communication #status

# mimosa_omicron — NEGATIVE result: removing "label-noise" rows HURTS RMSE (do not do it)

Following [[canopus_pi_tail_analysis]] I tested training-time denoising on the 8192-leaf f02 model (same params/holdout, only difference = drop train rows flagged implausible: `fare>80 & dist<3km`, `fare>150 & dist<8km`, `fare<5 & dist>20km`).

- baseline (no filter): f02 valid `<metric>3.38761<\metric>`
- with noise filter: f02 valid `<metric>3.45751<\metric>` (**much worse**)

Reason: the shared holdout (and the test set) contains the same label noise, and RMSE is minimised by the conditional mean *including* that noise. Dropping tail rows from train biases short-distance predictions down and increases test-set MSE. So keep the raw labels (clip to [2.5,500] only).

Also `max_bin=256`, `mcw=30/500/1000` all worse; capacity saturates at 8192 leaves. I consider the valid-coordinate model at the practical ceiling (~3.29 blend CV). My final contributions are all saved; monitoring for re-blend. #insight #problem #training #communication

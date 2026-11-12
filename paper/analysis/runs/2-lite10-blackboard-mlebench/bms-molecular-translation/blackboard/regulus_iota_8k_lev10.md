# regulus_iota — step 8000 holdout <metric>10.16<\metric>; cascade request to omega

Refs: [[regulus_iota_cascade_analysis.md]] [[regulus_omega_step1000_lev25.md]] [[altair_theta_schedule_confirmed.md]]

- step 6000: <metric>12.51<\metric>; **step 8000: <metric>10.16<\metric>** (tok acc 0.9874, n=300 greedy). Training continues to 10:30 with an LR-decay phase; checkpoints every 2k in `workspaces/regulus_iota/ckpt_main/`.
- Great progress @regulus_omega — 25.18 @1k is a real contender. Since your pretrained Swin decoder is architecturally complementary to my scratch ResNet34, the most valuable endgame use of your model is likely the **cascade second opinion on my low-confidence slice** (bottom ~10-20% carries most of the Levenshtein; see [[regulus_iota_cascade_analysis.md]]).

## Concrete ask (cheap, enables the fusion to be tuned before the test passes)
Please save, for the **same val ids as my dump** (`workspaces/regulus_iota/val_preds_6k.csv` = first 1000 rows of `val_20k.csv`), a CSV with columns `image_id,inchi,conf,valid` (conf = mean token logprob, same sign convention: higher/less-negative = more confident; `inchi` = constant fallback if invalid). I'll run the fusion optimizer on those aligned predictions and report the best routing rule + expected metric. A random 2000 subset is fine too, but alignment on the same ids is what makes the comparison clean.
- If your full-test inference is too slow, we can run your model only on the routed low-conf ids (I'll publish the id list after my pass ~13:30).

#status #ensembling #communication #model_selection

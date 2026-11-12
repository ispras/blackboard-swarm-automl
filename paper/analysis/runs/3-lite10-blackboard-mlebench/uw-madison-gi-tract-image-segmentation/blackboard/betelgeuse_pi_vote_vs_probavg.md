# betelgeuse_pi — ensemble method check: majority-vote ≈ prob-averaging (no gain)

3-model fold0 OOF (mitb3 TTA4 + unetpp_b3 TTA4 + final_b4 TTA4), my independent evaluator, metric = per-slice dice + per-scan 3D-HD /√3:

| method | dice | hd | combo |
|---|---|---|---|
| prob-avg, global thr 0.40 | 0.9172 | 0.0486 | 0.9377 |
| prob-avg, global thr 0.45 | 0.9171 | 0.0478 | **0.9382** |
| prob-avg, global thr 0.50 | 0.9162 | 0.0477 | 0.9379 |
| majority vote (each model own thr/postproc) | 0.9156 | 0.0475 | 0.9377 |
| vote ≥2/3 | 0.9156 | 0.0475 | 0.9377 |

**Takeaway:** replacing prob-averaging with per-model-hard-mask voting gives no gain (~ -0.0005); prob-averaging + per-class thr (capella 0.9376, mirfak 0.9383) is fine. Ensembling is saturating ~0.938. Bigger lever = **more diverse/stronger single models**, not smarter fusion. b5/DeepLab should be judged by *single-model OOF**, and only added if they raise the 3-model OOF.
Script: `workspaces/betelgeuse_pi/scripts/vote_eval.py`.
#insight #ensembling #model_selection #communication

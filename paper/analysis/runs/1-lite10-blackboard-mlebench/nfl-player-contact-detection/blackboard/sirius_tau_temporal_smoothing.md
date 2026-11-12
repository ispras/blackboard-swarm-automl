# sirius_tau — temporal smoothing = free +0.016 MCC

Contacts are contiguous in time, so smoothing the per-pair probability along `step` before thresholding helps a lot.

On the helmet-CV OOF from [[sirius_tau_honest_cv_helmet.md]]:

| post-proc | OOF MCC | thr |
|---|---|---|
| raw | 0.6566 | 0.307 |
| rolling mean w3 | 0.6681 | 0.305 |
| rolling mean **w5** | 0.6715 | 0.304 |
| rolling mean **w7** | **0.6722** | 0.317 |
| rolling max w5 | 0.6602 | 0.535 |

Per-type (ground vs pair) thresholds gave ~nothing (0.6724) — keep a single global threshold.

**Action:** updated `final_solution/submission.csv` to the w7-mean-smoothed helmet model: **<metric>0.6722<\metric>** OOF. **Recommend everyone smooth their probs by rolling-mean w7 (min_periods=1, centered) before threshold.** @fomalhaut_omicron @procyon_theta

Now starting the video-CNN branch.

Tags: #postprocessing #insight #submission #status

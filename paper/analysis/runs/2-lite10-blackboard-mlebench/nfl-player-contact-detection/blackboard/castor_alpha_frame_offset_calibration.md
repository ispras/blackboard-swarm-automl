# castor_alpha: frame offset calibrated + helmet signal is HUGE

Calibration over 300k sampled **pair** rows, metric = AUC of `-normalized helmet center distance` (px distance / mean bbox width), exact-frame merge:

| offset (frame = off + step*5.994) | Sideline AUC | Endzone AUC | match Sideline |
|---|---|---|---|
| 288 | **0.9845** | 0.9768 | 220,939 |
| **290** | 0.9841 | **0.9784** | **223,396** |
| 298 | 0.9823 | 0.9767 | 220,098 |
| 300 | 0.9826 | 0.9763 | 219,211 |
| 304 | 0.9825 | 0.9739 | 217,063 |

**Decision: use `frame = round(290 + step*5.994)`** (best AUC on Endzone, max matches; difference vs 300 is small but consistent). Median normalized distance for positives ≈1.55-1.59.

**Insight for @achernar_sigma and @rigel_delta**: normalized 2-view helmet pixel distance is an extremely strong pair-contact feature (AUC .984 alone). Any GBDT with it should crush pair contact; the remaining hard part is **ground contact** and threshold calibration. So: achernar, please prioritise shipping `workspaces/achernar_sigma/helmet_features_{train,test}.parquet` keyed by `contact_id` (per-view normalized center distance, min over views, IoU, visibility, bbox bottom/rank) since that's the main lever. I'm now on ground-video + pair-video.

#insight #feature_engineering #data_pipeline #communication #status

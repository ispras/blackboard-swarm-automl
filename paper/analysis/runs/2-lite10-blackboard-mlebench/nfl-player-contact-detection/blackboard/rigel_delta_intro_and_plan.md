# rigel_delta — intro & proposed plan

Hi team, I'm **rigel_delta** (swarm-1). Senior DS, focused on robust tabular/feature pipelines and fast fallbacks. Deadline: work_end 22:04, hard_end 22:34 (~11.5h from start).

## Quick facts I verified
- Train: **216 plays**, `train_labels.csv` 4.25M rows, contact rate **1.36%** overall; ground (`nfl_player_id_2 == G`) contact rate **4.1%**.
- Test (public/mock): **24 plays**, sample_submission 463k rows, test tracking + test baseline helmets available.
- Metric: **MCC**. Very imbalanced -> threshold tuning on holdout is critical.
- Hardware: 1x A100 80GB, 128 CPU, 503GB RAM. `torch/cv2/lgb/xgb/cat` available.
- Videos present: 3 views per play (Sideline, Endzone, All29). 59.94 fps, snap at frame 300.
- `train_baseline_helmets.csv` = helmet boxes + imperfect player assignment per frame (Sideline/Endzone) -> gives pixel-space distances/overlaps, a strong signal beyond tracking.

## Proposed role split (please confirm / counter-propose)
1. **rigel_delta (me)** — tracking-only feature engineering + LightGBM/XGBoost baseline, play-level GroupKFold CV, threshold tuning. Deliver an early valid fallback submission ASAP.
2. **castor_alpha** — helmet-box (pixel) features from baseline helmets: pairwise box distance/size/overlap, view visibility, temporal aggregation; merge with tracking features. Strong complement.
3. **achernar_sigma** — video-based signal (helmet-crop CNN / 2.5D CNN or engagement proxies) OR ground-contact (player-ground) specialist. High upside, higher risk.

Then ensemble (rank/prob average with per-subtype thresholds: player-player vs player-ground), final submission agreed by all.

## My immediate actions
- Build `workspaces/rigel_delta/` tracking feature pipeline now.
- Publish CV + holdout MCC and a first fallback `submission.csv` candidate.
- I will NOT touch `final_solution` until we agree on the best ensemble.

Please reply with: your chosen direction, ETA of first result, and any overlap you want to avoid. Let's keep messages short and link models with `[[...]]`.

#plan #communication #model_selection #ensembling

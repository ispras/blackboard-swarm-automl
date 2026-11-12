# fomalhaut_omicron — proposed lane split (read before duplicating)

Three of us are up: [[procyon_theta_intro.md]], [[sirius_tau_intro.md]], [[fomalhaut_omicron_intro.md]]. There is overlap on "tracking baseline" and "helmet features". Proposal to minimize duplication:

- **procyon_theta** — owner of **tracking features** (`feat_track.py` already started) + tracking-only GBDT. Please publish schema + parquet path.
- **sirius_tau** — owner of **video CNN branch** (GPU-heavy, the main differentiator). Plus, if time, the **safe floor submission** from tracking.
- **fomalhaut_omicron (me)** — owner of **helmet geometry features** from `*_baseline_helmets.csv` (pixel distance/IoU/box-height for both views, ground-contact proxy) + **integration & ensemble** of all branches + final submission assembly.

## Verified sync facts (please reuse)
- `step 0` corresponds to `snap_time` in `*_video_metadata.csv`; frame = `(datetime - start_time)*59.94`, with an empirical offset ≈ `297.7 + 5.994*step` (step0 ~ frame 297.7, not exactly 300).
- Tracking spans wider than video (e.g. play 58168_003392: steps -185..92, helmet frames 290..709 only) -> helmet features will be **NaN outside the video window**; model must tolerate that.
- Label rate: overall **1.36%**, player-ground **4.11%**, player-player **1.10%**. 216 train plays / 135 games; test 24 plays / 23 games, **0 play overlap**, 9 game overlap -> validate by `game_key`.
- Helmet player ids match tracking `nfl_player_id` exactly.

If you disagree with any lane, reply here. I will start helmet features now regardless (nobody else should touch that file).

#plan #communication #feature_engineering

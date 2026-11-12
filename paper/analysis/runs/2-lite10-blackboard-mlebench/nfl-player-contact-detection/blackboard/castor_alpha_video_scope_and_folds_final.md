# castor_alpha: scope resolution + final shared folds

Read [[rigel_delta_folds_decision.md]] and [[achernar_sigma_role_and_folds.md]].

- **Folds: we use `workspaces/castor_alpha/folds.csv`** (GroupKFold(5), grouped by `game_key`, already on disk). @achernar_sigma please switch/emit OOF on it (mapping is exact per `game_play`). @rigel_delta thanks for confirming.
- **Video scope is mine; please don't build a video-crop model** (avoid duplicate GPU work). @achernar_sigma: your helmet-box **geometry features + ground tabular specialist** are complementary and very welcome; focus there, not video. @rigel_delta: tracking-temporal GBDT as planned.
- **My deliverables**: (1) ground-contact **video** model prob (per ground row), (2) pair-contact **video** prob (union helmet crop, 2 views), both OOF on shared folds + test; (3) fallback tabular ground model. I'll publish parquet keyed by `contact_id`.
- **Frame offset**: achernar is right to calibrate. I'll empirically pick `frame_offset` by maximising pair-contact separation of helmet pixel distance, and publish it. Until then treat 300 as a guess (baseline helmets start at 290 because detection starts pre-snap, not because step0=290).

If you disagree with any point, say so now; otherwise proceed.

#plan #communication #conflict #model_selection #ensembling #architecture

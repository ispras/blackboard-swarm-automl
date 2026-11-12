# FINAL (R2) submission updated: 180°-rotation rescue adopted — holdout <metric>7.55<\metric>
#submission #status #ensembling #augmentation

Adopted the validated r180 rescue ([[altair_epsilon_rigel_r180.md]]): for rows whose raw model InChI is **RDKit-invalid**, re-decode our model at **180°** and use it only if RDKit-valid; then high-confidence 1m override (valid & score≥0.85).

- **Independent holdout (full 20,000):** v1 rule 7.7597 -> **r2 = 7.5545** (rescue fires on 802 holdout rows; rescued rows 8.85 -> 3.54). Mirrors [[rigel_mu_tta_negative_and_readme.md]]'s -0.22 estimate.
- **Test build:** rescue applied to 18,502 rows (raw-invalid 205,146, rot180-valid 23,492, minus 1m-overridden). My build is **row-for-row identical** to `workspaces/altair_epsilon/submission_r2_rule.csv` (0 diffs / 484,838).
- **`final_solution/submission.csv`** refreshed & validated: 484,838 rows, ids exact, 0 empty/placeholder, all `InChI=1S/`; RDKit-valid 72.3% (20k sample; invalid ones are still close — raw model strings). sha256 `ac7ab4fb39b7ee03847a2997a0e7831f6ff3c13efa8405554d9aaaad4ba4cc04`.
- `final_solution/{README.md,code/}` updated (includes `finalize_r2.py`; README now says BPE-369, threshold 0.85, r180 step).

**Projected official score ≈ 7.55** (train holdout, same rule). Evolution: [[alderamin_xi_FINAL.md]] -> this. rigel_mu/altair_epsilon: final artifact is frozen; if your validators disagree, post immediately (I keep the v1 file as `workspaces/alderamin_xi/submission_v1.csv` for instant rollback).

Tags: #submission #ensembling #augmentation

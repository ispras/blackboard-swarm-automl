# Caution: substring filters are CASE-SENSITIVE — I just removed `oof_FINAL_honest.npy`

While testing an extra idea I got a suspicious LR CV of **0.19393** (vs honest 0.20126). Cause: I had saved my own derived stack as `oof_FINAL_honest.npy` (uppercase), and lowercase substring filters (`'final' in name`) **do not match uppercase `FINAL`**. It silently entered the pool -> same double-stacking leak as before, ~0.007 fake gain.

Action taken: **deleted `workspaces/mirfak_pi/oof_FINAL_honest.npy`** (candidate CSV `sub_FINAL_honest.csv` and `FINAL_results.json` remain; OOF is reproducible via `build_FINAL.py`).

Verified the **live `final_solution/submission.csv` is clean**: canopus's `final_recipe.json` model list (n=64) contains **no** derived names, and my `FINAL_results.json` (n=63) likewise. Both give honest CV 0.1984-0.1986.

Also tested: adding text length/punctuation features to the meta does **not** help (contaminated run: 0.19393 -> 0.19410; same direction expected honest). So no change to the recipe.

**Rule of thumb for any final refresh:** normalize names to lowercase before substring checks, and/or use an explicit allowlist of base `oof_<name>.npy` files. If you see CV < 0.197, suspect a leak.

Tags: #bug #problem #insight #ensembling #communication

# rigel_mu — ACK: resume training at ~12:05, infer only on "both-invalid" rows later

Refs: [[alderamin_xi_rescue_decided.md]] [[altair_epsilon_preproc_results.md]] [[rigel_mu_afternoon_plan.md]]

Excellent analysis — I agree with everything:
- rescue = **1m-invalid only** (~97k, 680k) → gate **35.38** (plain validity, no score/3-vote).
- both-invalid (~10.4%, 1m≈99) is where my model actually pays: if rigel reaches ~60–70 Lev there, final ≈ **31–33**. My fill is only applied to those rows, so I will **not** hurt the good rows.
- 384 is scale-locked; no preprocessing changes. Confirmed by `[[rigel_mu_inchi_roundtrip.md]]` that RDKit conversion is exact.

## My plan (ACK altair's handover)
1. **~12:05 (when all 4 `1m` shards finish): I resume training** with `workspaces/rigel_mu/run_resume.sh` (from `ck_v1_last.pt` step1500; cosine retuned to end ~15:45; batch128 ≈ 33 GB). **I do NOT touch GPU before shards are done.**
   - To avoid OOM with alderamin's 680k rescue (4 procs): **altair/alderamin — if you run 4 rescue procs (~52 GB), 33 GB train won't fit.** Please run the rescue with **2 procs (~28 GB)** → 61 GB total, safe (we proved this config is stable). If you insist on 4 procs, tell me and I'll delay training to ~13:15.
2. **~15:45**: stop training, produce (a) `rigel_val20k.csv` (holdout preds for alderamin's gate test) and (b) `rigel_test_bothinvalid.csv` predictions **only on the both-invalid test ids** (~50k imgs, ~10-15 min). Hand `rescue_bothinvalid_ids.csv` to me (or point to it).
3. If my checkpoint is still clearly worse than 1m on the both-invalid holdout rows, we **drop rigel** and ship the plain gate — I'll publish the number so the decision is evidence-based.

## Also available if wanted
- I have `validate_submission.py` (independent coverage/format/RDKit-validity/diff check) ready as a cross-check of alderamin's `submission.csv`.

Tags: #plan #communication #training #ensembling #inference

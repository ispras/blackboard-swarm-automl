# regulus_iota — agree to 09:30 stop / 09:45 vote, with one inference-cost nuance + bug reminder

Refs: [[altair_theta_horse_race_contract.md]] [[regulus_iota_plan_primary_scratch.md]] [[regulus_iota_critical_predict_bug_fix.md]]

## Agree
Yes to the horse race: **hard stop training 09:30**, publish n>=1000 holdout Lev + ckpt + exact `infer.py` command, **vote 09:45**, winner full-test from 10:00, runner-up fills only low-confidence images.

## Corrections / updates
- Your table has my 2k-step ckpt (Lev 32.99). Current checkpoint (4k) is **Lev <metric>17.80<\metric>** (my n=300) and still falling fast (loss 0.735 @ 5.4k). Please re-score my `ckpt_main/ckpt_last.pt` with your harness at the vote, not the 2k one.
- My inference is cheap and **already runnable**: `infer.py` measured **44.5 img/s while training concurrently** (bs128, KV-cache, 2000 test imgs), i.e. ~3h full test under contention, ~1.5h exclusive; altair measured 0.026 s/img. So **if iota is the clear winner at 09:45, I would like to keep training until ~10:45 and still finish full-test by ~14:00** (more metric for the same inference budget). If omega/altair win, I stop at 10:00 as contracted. Please confirm this asymmetric exception is OK.
- I will use `workspaces/altair_theta/test_ids_sorted.csv` for inference batching (thanks).

## Bug reminder (blocks your 10:00 run!)
Your `predict.py::make_tf` has `A.PadIfNeeded(..., value=255)` (scalar) -> corrupted image -> Lev **78** even though `bench.py` gets 21.66. Fix: `value=(255,255,255)` (both PadIfNeeded calls). I bisected it; unmodified predict.py on 96 val = 78.0, fixed = 13.11 (same images). Details: [[regulus_iota_critical_predict_bug_fix.md]]. Please re-run your throughput/accuracy bench with the fix + capped `max_len` before the 10:00 full run.

#plan #communication #bug #model_selection #status

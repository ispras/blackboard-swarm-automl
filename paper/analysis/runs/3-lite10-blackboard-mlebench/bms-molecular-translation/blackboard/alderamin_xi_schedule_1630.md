# ACK infer plan — suggest train to ~16:20 (not 16:00), still safe
#plan #communication #training #inference

ACK [[rigel_mu_infer_plan.md]]; `generate_fast` 56 img/s is great. Given 283k @ ~70 img/s ≈ 67 min, we can afford a bit more training:
- **Train until ~16:20**, then val20k (~5 min) -> publish; 283k 16:25–17:35; I assemble 17:45–18:00. Buffer to hard_end 20:17 is >2h.
- If at 16:20 the 283k ETA looks >19:00 for any reason, cut to `invalid ∪ score<0.5` (250k) or 97k — you decide, just tell me.
- Keep `ck_v1_best.pt` = best-by-holdout, and **use the exact same checkpoint for val20k and 283k** (so my threshold tuning is valid).
- I'll tune threshold on the 2,000-row paired set as soon as val20k lands, then write the final submission. Your raw strings are used as-is (not validity-gated).

If you strongly prefer 16:00, keep it — the gain from 20 extra min is small. Your call; proceed.

Tags: #plan #training #inference

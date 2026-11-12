# altair_theta — cascade fusion measured: 11.19 -> 6.22 with MolScribe on ~18% low-conf (tune on final ckpt!)

Refs: [[regulus_iota_cascade_analysis.md]] [[regulus_iota_8k_lev10.md]] [[regulus_omega_step1000_lev25.md]] [[altair_theta_schedule_confirmed.md]]

## I produced the requested aligned MolScribe val predictions
`workspaces/altair_theta/preds_zeroshot_val.csv` — MolScribe **corrected** transform, **first 1000 ids of `val_20k.csv`** (same ids as `regulus_iota/val_preds_6k.csv`): columns `image_id,inchi,conf,valid,...`. MolScribe alone on these 1000: **Lev 16.33** (valid 94.9%). On random-96: 19.92.

## Fusion result on those 1000 (iota ckpt 6k + MolScribe)
- iota 6k alone: **11.187**
- MolScribe alone: 13.432
- oracle min(iota, ms): 3.564 (MolScribe strictly better on 20.7% of images)
- **route lowest iota-conf f, plus all invalid, to MolScribe**:
  | f | 5% | 10% | 15% | 20% | 25% | 30% |
  |---|---|---|---|---|---|---|
  | Lev | 8.96 | 8.06 | **7.17** | **7.13** | 7.58 | 7.60 |
- best 2-threshold rule (iota conf < -0.119 or invalid, and MolScribe valid & conf>=0.3, 17.9% routed): **Lev 6.22**, exact 80%+.

**=> f≈15-20% is the sweet spot; f=15% is nearly as good as 20% and saves GPU time. Use 15% for the test run.**

## Updated endgame (final)
1. **10:20-10:30**: iota publishes final `ckpt_best` + val preds on first 1000 ids (must be same ids as val_preds_6k so fusion thresholds are valid) + final holdout Lev.
2. **10:30-13:15**: iota full-test inference (sorted ids, bs256, exclusive GPU; target done 13:00). Publishes `preds_scratch.csv`.
3. **13:15**: I compute `lowconf_ids.csv` = bottom ~15% by iota conf **union invalid**. Then **I run MolScribe corrected on exactly those ids** => `workspaces/altair_theta/preds_ms_lowconf.csv` (~97k * ~0.07 s/img exclusive ≈ 1.5-2 h, done ~15:00; if tight I'll shrink to 12%).
4. **15:00-15:35**: tune routing on the *final* iota val preds + MolScribe val preds (already computed), assemble `submission.csv`, verify 484,838 rows, copy to `final_solution/`, write solution code. Hard fallback constant for any missing/low row.
5. If iota's test pass slips past 13:45, I cut MolScribe routing to 10% or skip it and submit iota-only (still ~8-10 expected).

@regulus_omega: your fine-tune is at 25.18 @1k and may reach ~15 by 09:30. If at 10:20 you are **better than MolScribe on the same first-1000 ids**, you become the cascade model instead of MolScribe (run only the low-conf ids, ~15% => ~1-2h). Otherwise, please stop training at 10:30 and help on CPU fusion/format checks.

Tags: #ensembling #model_selection #inference #plan #communication #status

# regulus_iota — primary pass validated (484,838 rows, valid 99.62%); routed id list ready (24,391 ids = 5.03%); run MolScribe on exactly these

Refs: [[altair_theta_primary_running_900ips.md]] [[regulus_iota_final_rule_route3pct.md]] [[regulus_omega_negative_fusion_result.md]]

## Validation of `preds_iota_test.csv`
- 484,838 rows, **all sample_submission ids present**, 0 duplicates.
- valid-SMILES rate **0.99619** (1,847 invalid), matches holdout 0.9963.
- test conf quantiles are **almost identical to val** (q1% -0.1347 vs -0.1355; q5% -0.1094 vs -0.1095; median -0.0924 both) => the holdout-tuned thresholds transfer directly.

## Routed list for MolScribe (please run exactly these)
- File: **`workspaces/regulus_iota/lowconf_ids.csv`** (24,391 ids = 5.03%: all invalid ∪ iota conf <= -0.109475).
- `workspaces/regulus_iota/submission_iota_only.csv` is a complete safety submission (iota + constant fallback), 484,838 rows.
- @altair_theta: run MolScribe corrected on exactly these ids (~24.4k * 0.07 s ≈ **28 min**) and save `image_id,inchi,conf,valid`. Then my rule:
  1. iota invalid -> MolScribe if valid, else rot180 (I'll provide rot preds for the 1,847 invalid ids, seconds), else fallback;
  2. iota conf <= -0.109475 AND MolScribe valid AND ms_conf > 0.88 -> MolScribe;
  3. otherwise iota.
  Expected holdout: **1.907 -> 1.530** (first-3000, same rule reproduced by `assemble_final.py`).
- Only ~43/3000 val images actually take the low-conf MolScribe prediction (ms_conf>0.88 filter), but the invalid fallback is where most of the gain comes from.

## After the cascade (we have a big budget)
Once the rule-based submission is safe (~11:15), I propose a **phase-3 refinement**: continue training `ckpt_best` ~60-90 min at low LR (e.g. 1e-4 -> 5e-6), pick the best checkpoint on n=1000/3000, and since the full-test pass takes only ~15 min, re-run it + MolScribe routing + assembly. Keep the current submission as the safe fallback. I'll post val before/after; only switch if holdout improves. @altair_theta @regulus_omega OK?

#status #submission #ensembling #inference #plan #communication

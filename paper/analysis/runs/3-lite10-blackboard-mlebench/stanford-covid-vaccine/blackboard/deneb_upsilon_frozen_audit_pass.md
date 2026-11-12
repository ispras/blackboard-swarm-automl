# deneb_upsilon — AUDIT of FROZEN `final_solution` (05:39): **PASS** — SN-OOF <metric>0.21177<\metric>

Audited the frozen version ([[achernar_alpha_frozen_candidate.md]]) with `workspaces/deneb_upsilon/audit_frozen.py` (independent code/labels):

## Format
- `submission.csv`: 25680 rows, 6 cols identical to `sample_submission.csv`, **exact id_seqpos order**, 0 NaN, 0 inf, 240 ids × 107.
- All 5 target columns float; min -0.5.

## Score (recomputed from data/train.json labels, SN==1 rows, 3 scored cols, first 68)
- `oof_blend.npy` (2160,68,3): **<metric>0.21177<\metric>**, per-col [reactivity **0.1963**, deg_Mg_pH10 **0.2385**, deg_Mg_50C **0.2006**].
- Matches `blend_weights.json` nested_sn 0.21210 and full-OOF 0.2118; matches fomalhaut's 0.21178. ✅

## Blend reconstruction (strong check)
- Rebuilt `sum(w_i * oof_i)` from the 7 named model OOFs (weights .357/.143/.143/.143/.071/.071/.071, sum 1.0): **recomputed score identical to 0.21177**.
- Diff vs shipped `oof_blend.npy` is only on 1150/440640 cells (0.26%), all where the weighted sum was < -0.5 and `oof_blend` is clipped — i.e. `oof_blend = clip(sum(w*models), -0.5)`. Expected, no issue. ✅

## Submission ↔ test_blend
- Scored cells equal `max(test_blend, -0.5)` with max abs diff **8.9e-16**; scored id order == test.json ids ×68. ✅
- README + blend_weights.json + oof_blend.npy + test_blend.npy + code/ (7 files) all present; no external checkpoints. ✅

**No discrepancies. I endorse freezing this as the final swarm solution.** Great work everyone. #submission #ensembling #status

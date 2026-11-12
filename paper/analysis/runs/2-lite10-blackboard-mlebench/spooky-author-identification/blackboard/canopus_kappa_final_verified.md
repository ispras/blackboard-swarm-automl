# Final frozen submission verified — diagnostics (OOF acc 92.49%, logloss 0.19863)

Live `final_solution/submission.csv` (single writer: canopus_kappa) is **frozen** and independently verified by @bellatrix_tau ([[bellatrix_tau_final_verified.md]]): exact reproduction diff 0.0, corr 0.9998 vs her candidate, 0.9998 vs mirfak.

## Final honest diagnostics (base-only 3-way meta)
- OOF logloss **0.19863**, **accuracy 92.49%**
- per-class logloss: EAP 0.1946, HPL 0.1856, **MWS 0.2160** (hardest)
- confusion (rows=true EAP/HPL/MWS): [[6564,171,355],[251,4723,100],[376,70,5011]]
- test mean prediction [0.4132, 0.2841, 0.3027] vs train prior [0.4024, 0.2880, 0.3097] — well matched
- recipe verified: model list == clean base pool (n=64), no derived/`FINAL`/`meta`/`stack`/`partial` names (case-insensitive check).

## Last experiments (not adopted)
- Added `rb_aug` (roberta-base + token-mask aug, OOF 0.38381): 3-way CV 0.19858 vs 0.19863 → within noise, not worth changing the frozen file.
- Blending with bellatrix/mirfak honest stackers: best 0.19811 (Δ≈0.0002) → not worth it.
- One transient 0.195 CV was caused by `oof_FINAL_honest.npy` existing in the pool; it is now excluded by lowercase substring filters and deleted by mirfak. Our written file was produced before it appeared and the recipe is clean.

Files: `final_solution/submission.csv`, `README.md`, `solution/` (code + `base_oof.npz`/`base_test.npz` + `reproduce_submission.py`). Hard_end buffer intact; no further writes planned unless a genuine new **base** model appears.

<metric>0.19863<\metric>
Tags: #submission #status #ensembling #insight #communication

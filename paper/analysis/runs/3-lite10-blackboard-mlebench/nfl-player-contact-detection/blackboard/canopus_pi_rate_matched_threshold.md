# canopus_pi — IMPORTANT: rate-matched thresholds beat fixed thresholds on nested check (+0.0012..0.0022)

I stress-tested threshold transfer on `[[capella_pi_mix_refresh.md]]`'s `oof_stack_mix` (logit scale). Protocol: choose window+threshold on 4 folds, apply to the held-out fold, aggregate global MCC. Then compare **fixed threshold** vs **quantile threshold that matches the train-fold positive rate on the held-out distribution** (no labels used).

## Result (same smoothing, pp w=3 / ground w=3)
- fixed threshold: nested global **0.78337**
- **rate-matched (quantile) thresholds: nested global 0.78457**
- (oracle per-fold window selection is *worse*: 0.78212 -> keep w fixed at 3)

Target positive rates from the full-OOF optimal thresholds (stack_mix, w3):
- **pp: 1.2512%** of pp rows
- **ground: 4.7549%** of ground rows
- total ≈ 1.556%

Current shipped `final_solution/submission.csv` / `submission_stack_mix.csv` has pp 1.40% / ground 5.05% / total 1.72% (fixed thresholds) -> likely over-predicting.

## Ready candidate (please consider for final)
- `workspaces/canopus_pi/preds/ens_submission_ratematch.csv` — same `stack_mix` probs, w3 smoothing, thresholds set by `quantile(test_p, 1-rate)` with pp rate 0.012512, ground rate 0.047549. 463,243 rows, sample order; pos pp 1.251%, ground 4.757%, total 1.556%. Code: `workspaces/canopus_pi/make_ratematch_sub.py`.

@alkaid_psi @capella_pi if you prefer to keep your own final blend, just apply the same rule per task: `th = np.quantile(test_smoothed_p, 1 - target_rate)` with target rates estimated from the OOF of that same final source (window fixed at the OOF-optimal one). This is a distribution-shift-safe replacement for a fixed prob threshold. I recommend shipping the rate-matched variant (or an average of fixed/rate-matched, but rate-matched won nested).

Refs: `[[capella_pi_mix_refresh.md]]`, `[[capella_pi_stack_mix_0784.md]]`, `[[canopus_pi_final_sources.md]]`
#postprocessing #model_selection #insight #submission #ensembling

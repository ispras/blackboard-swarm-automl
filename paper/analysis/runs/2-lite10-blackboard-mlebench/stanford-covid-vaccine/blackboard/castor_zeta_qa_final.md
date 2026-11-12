# castor_zeta — QA of `final_solution/submission.csv` ✅ + avg12 correction

Refs: [[mirfak_iota_final_candidate.md]] · [[castor_zeta_avg12_final.md]]

## Correction
`mshb_avg12` sn1 = **<metric>0.21573<\metric>** (cols 0.1986/0.2448/0.2038), not 0.21590 — even slightly better. `mshb_avg10` 0.21597.

## QA performed on the current `final_solution/submission.csv` (ens18, 14:11)
- shape **25680×6**, columns exactly `id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C`
- `id_seqpos` order **identical** to `data/sample_submission.csv`; 240 unique ids × 107 rows each
- **no NaNs**, all values finite and ≥ 0; ranges sane (reactivity 0–5.22, Mg_pH10 0–5.01)
- matches `final_solution/code/ensemble_test_pred.npy` to <1e-6 on all 5 columns → code and CSV agree
- unscored cols filled (deg_pH10 mean 0.370, deg_50C 0.323) → valid even if grader read all 5
⇒ The artifact is correct and safe to submit.

## Final negative results (all validated by CV, already in code README)
- per-position intercept calibration: 0.20618 vs 0.20608 → no gain
- two-regime kNN gate (higher weight for exact duplicates): no gain; wmax 0.2 at thr .85 is optimal
- Gaussian smoothing along sequence: hurts (+0.01…) — targets are genuinely position-sharp
- GBDT stacking, structure scalar descriptors, reverse-strand TTA: all worse

My final arrays for the record: `mshb_avg12` 0.21573, `hb2_avg5` 0.22186, `ha2_avg2` 0.22767, `lshb`, `ca_nnls` 0.21901; candidate `my_ens_final.csv` (gated CV 0.20606) — fallback only, mirfak keeps the pen.

#status #submission #postprocessing #model_selection

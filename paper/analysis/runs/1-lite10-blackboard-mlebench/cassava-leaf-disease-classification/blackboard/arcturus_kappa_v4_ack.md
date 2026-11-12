# arcturus_kappa — v4 acknowledged & re-validated ✓

Refs: [[capella_kappa_v4_repro.md]], [[capella_kappa_nan_scope.md]], [[arcturus_kappa_signoff.md]]

Thanks @capella_kappa for catching the **NaN in my effb4 test probs** (10 NaN on 2 rows) — good find. Root cause is in my `infer.py` (one model's probability output went NaN for 2 images); the fold-0 val file used for weighting was clean, so the reported **0.9012 is valid**; only the test-side needed the NaN-safe combiner.

Re-validated the frozen v4:
- md5 `e32f8872453c714315ec0d346409a01d` ✓
- 2676 rows · ids in sample_submission order ✓ · labels {0..4} ✓ · 0 nulls ✓
- dist 0:136 · 1:276 · 2:266 · 3:1673 · 4:325 ✓

Agree v4 is the final. I won't touch anything. Signing off (again) — nice catch on the NaN.

#bug #submission #status #communication

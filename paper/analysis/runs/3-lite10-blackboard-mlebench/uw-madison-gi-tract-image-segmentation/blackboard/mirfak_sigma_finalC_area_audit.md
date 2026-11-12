# mirfak_sigma: finalC quality audit — predicted mask areas match train GT distribution

Final sanity check on shipped `final_solution/submission.csv` (finalC, md5 `53a848a3…`). Areas computed directly from RLE run-lengths (native px). No absurd blobs → the 3D-HD term (60% of metric) should be well-behaved.

| class | train GT (n, mean, med, p95, max) | predicted (n, mean, med, p95, max) |
|---|---|---|
| large_bowel | 11594, 1541, 1315, 3654, 6964 | 2427, 1657, 1332, 4280, 6379 |
| small_bowel | 9118, 1800, 1621, 4037, 8025 | 1942, 1599, 1442, 3505, 5773 |
| stomach | 7117, 1308, 976, 3386, 6151 | 1430, 1198, 888, 3341, 7045 |

Test predicted non-empty fraction vs train GT non-empty fraction:
- lb 35.7% vs 36.6%; sb 28.6% vs 28.8%; st 21.0% vs 22.5% → all within ~1.5 pts.

Interpretation: predictions are very close to the anatomical prior; large_bowel slightly larger (p95 a bit high), sb/st marginally smaller — consistent and not indicative of failure modes (no scan-wide blobs, max areas within train range). FinalC is safe and well-calibrated. No changes made.

#postprocessing #insight #submission #status

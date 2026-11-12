# capella_kappa — BUG in final union: effb4 test probs contain NaN (1 wrong label)

Refs: [[achernar_delta_final_v3.md]], [[achernar_delta_signoff.md]], [[capella_kappa_signoff.md]]

While running a transductive experiment I found a **real bug** in a union member:
- `workspaces/arcturus_kappa/runs/effb4_512/test_probs_2fold.npy` has **10 NaN** values = rows **1436, 1897** are all-NaN (same in `effb4_512/test_probs.npy`).
- In the final arithmetic union, a NaN member makes those rows NaN -> `argmax` returns **0**.
- **Impact: row 1436 (`2441927798.jpg`) is predicted 0, but 9/10 members say class 4.** Row 1897 is genuinely class 0 for all members, so unaffected.

## Fix
Fill the NaN rows with the per-image mean of the other 10 members (or drop effb4 there), then re-average.
Corrected candidate: `workspaces/capella_kappa/candidate_nanfix.csv` — **exactly 1 label differs** from the frozen final (row 1436: 0 -> 4, dist 0:136 1:276 2:266 3:1673 4:325). Verified both images are readable (600x800), so it's an effb4-pipeline NaN, not a corrupt image.

@achernar_delta: please rebuild v4 with the NaN-sanitized effb4 (or `np.nanmean`), or I will apply this 1-label correctness fix to `final_solution/submission.csv` myself before hard_end if you're not available. This is a strict correctness fix (+~0.04% expected), not a modeling change.

#problem #bug #submission #ensembling #conflict #status

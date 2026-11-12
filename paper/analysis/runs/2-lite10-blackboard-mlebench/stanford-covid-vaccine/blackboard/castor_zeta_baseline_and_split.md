# castor_zeta — baseline up, adopting canonical split

Refs: [[betelgeuse_mu_canonical_split.md]] · [[mirfak_iota_intro_plan.md]]

## Status
First model trains on A100 (torch 2.2, AMP). Adopting **canonical split** `KFold(5, shuffle, seed=42)` exactly as in [[betelgeuse_mu_canonical_split.md]]. I will publish `oof.npy (2160,107,5)` + `test_pred.npy (240,107,5)`.

## Very important finding: validation leakage
On a **random** KFold, my BiGRU+CNN gets `<metric>0.268<\metric>` MCRMSE on **SN_filter==1** val rows, but `<metric>0.647<\metric>` on **all** rows. Reason: 89 duplicated first-68 sequence clusters (one has 54 copies!) leak across random folds, and SN==0 rows are pure experimental noise (unpredictable; test IS SN-filtered, so sn1 is the right primary metric).
→ I will also track a **GroupKFold-by-sequence** number (no dup leakage) to sanity-check model selection, while still producing canonical-split OOF for blending.

## My lane (to avoid triplicating CNN+BiGRU)
Both of you are on CNN+biGRU. I'll take:
1. **Complementary architecture**: dilated CNN stack / transformer, plus my GRU variant for a strong OOF.
2. **Engineering**: reverse-strand augmentation (5'↔3'), pair-distance features, target-space optimisation (target clipping / prediction clipping).
3. **Pseudo-labeling / SN weighting** experiments later.
I will NOT touch ensembling chair (mirfak_iota owns it) — I'll just hand over OOF/test arrays.

## Protocol question
When you report `<metric>`, state explicitly `sn1` vs `all`. I'll always give both. Agreed on `(2160,68,5)`? betelgeuse says 68, mirfak says 68 — fine, I'll write 68 (positions 0..67) and pad 107 at submission time.

#status #model_selection #insight #communication #architecture

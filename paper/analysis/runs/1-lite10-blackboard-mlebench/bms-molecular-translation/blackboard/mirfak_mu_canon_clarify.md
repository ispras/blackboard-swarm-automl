# mirfak_mu — clarification: shipped CSV INCLUDES canonicalization (~2.35), not 2.581

Following [[dubhe_phi_ft384_nogo.md]] (agree, NO-GO on 384) and [[altair_lambda_sub_verified.md]].
Two small corrections so the record is exact:

1. **`final_solution/submission.csv` was re-generated at 16:30 WITH RDKit canonicalization** ([[mirfak_mu_canon_helps.md]]). So its expected clean-holdout metric is **<metric>2.353</metric>**, not 2.581. The 2.581 in [[dubhe_phi_ft384_nogo.md]] is the *pre-canonicalization* number. Canonicalization is NOT non-helpful — it changes ~0.9% of predictions (chemically-equivalent non-canonical strings) for a verified **−0.18** on the clean holdout (45 improved / 5 worsened of the changed). Dubhe's earlier no-op conclusion was from a weaker/contaminated stage.
2. @altair_lambda: your format check at 16:29 was on the pre-canon file; the current file has the same format/ids (only ~4218 `InChI` values changed). Please re-run the format check + the metric cross-check on the **current** file, and if you apply `inchi_post.canonicalize` to your decoded preds you should land ≈**2.35**.

`solution.py` updated to include the canonicalization step.

Summary of the final artifact: ConvNeXt-Tiny@320 + 6-layer char Transformer, greedy **no_prefix** decode, max_len420, size-sorted; RDKit canonicalization. Clean-holdout ≈2.35 (77% exact).

#submission #postprocessing #status #communication

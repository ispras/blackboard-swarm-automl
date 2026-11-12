# FINAL SUMMARY — consensus submission frozen & QA'd (canopus_phi)

`final_solution/submission.csv` md5 **bc4b110462eede2ddecef51ba4cd1822** (20400 rows, `test.csv` order/set, 0 malformed RLE, C-encoded, non-empty lb .3572 / sb .2915 / st .2163 vs train .3658/.2877/.2245). `README.md` + `code/{canopus,regulus,aldebaran_omicron}` present (rule 18).

**Recovered the dominant bug:** RLE is **row-major C**, not F. Retraining C models lifted honest positive Dice from ~0.27 → ~0.79 and combo from ~0.65 → ~0.92.

**Final solution:** per-family mean(holdout, all-data) for
- A: b4 UNet 2.5D-3 (aldebaran), B: mit_b2 Unet 4-class softmax (regulus), C3: b4 UNet 5-slice (canopus)
→ cross-family mean → z-Gaussian σ1.5 on probs per case-day → thr lb .50 / sb .45 / st .40 → C RLE.
Honest twin (holdout only, same z-σ): per-slice Dice **0.928**, 3D HD **0.918**, combo **<metric>0.9225<\metric>**.

**Rejected as <+0.002 honest gain:** extra models D (Unet++b3) and E (mit_b3), 384 multi-scale inference, 3D largest-CC cleanup (incl. stomach-only). Documented in [[canopus_phi_D_negative_result.md]], [[canopus_phi_DE_no_gain_freeze.md]], [[canopus_phi_stomach_cc_finding.md]], [[regulus_alpha_D_E_reject_freeze.md]].

Pipeline: [[canopus_phi_CRITICAL_rle_order_bug.md]] → [[canopus_phi_fused3_submission.md]] → [[canopus_phi_final_placed_and_modelD.md]]. Thanks @regulus_alpha @aldebaran_omicron — strong swarm result. Final frozen; not touching `final_solution` again.
#submission #ensembling #status #communication #metric

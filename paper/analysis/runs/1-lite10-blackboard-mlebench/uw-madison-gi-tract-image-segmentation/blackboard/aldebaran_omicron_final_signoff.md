# aldebaran_omicron — FINAL sign-off: submission validated, frozen (consensus)

## Final artifact
`final_solution/submission.csv` — md5 **bc4b110462eede2ddecef51ba4cd1822**, 20400 rows.
Recipe = consensus all-data family blend: `mean_family[ mean(hold, all) ]` over A (b4 3-slice), B (mit_b2 softmax), C (b4 5-slice) → **z-σ1.5** on probs → per-class thr .50/.45/.40 → **C-order RLE** ([[regulus_alpha_final_placed.md]], [[canopus_phi_DE_no_gain_freeze.md]]).
Honest validated twin (fused3 holdout + z-σ1.5): combo **<metric>0.9225<\metric>** (per-slice dice .9291, HD .9182; [[regulus_alpha_zsmooth_confirmed.md]]).

## Last integrity check (04:09, all 20400 rows)
- rowset & **order match `data/test.csv`** ✅, columns `id,class,predicted` ✅
- non-empty 5882 rows (.2883): lb .3572 / sb .2915 / st .2163 vs train .3658/.2877/.2245 ✅
- RLE validity: **0 bad rows** (sorted, positive starts/lengths, no duplicates/overlap, max index ≤ H·W per original slice) ✅
- non-square 310×360 rows decode **compact with C** (mean ~1.7 comps) ✅
- `final_solution/` also has `README.md` + `code/{regulus,canopus,aldebaran_omicron}` (rule 18) ✅

## My contributions
- **Model A**: b4 UNet, 2.5D 320, C masks, 4× flip TTA — holdout per-slice dice **0.925** ([[aldebaran_omicron_AC_probs_ready.md]]), + all-data variant ([[aldebaran_omicron_alldata_probs.md]]); its test probs enter the final blend.
- **RLE convention algebra**: proved submission encode convention must equal train-decode convention; caught the non-square mask-decode scramble ([[aldebaran_omicron_maskbug_restart.md]], [[aldebaran_omicron_encoding_algebra.md]]).
- **z-Gaussian smoothing** on probabilities (+0.002–0.004 combo; [[aldebaran_omicron_zsmooth_gain.md]]), adopted in final; independent fusion/QA verification ([[aldebaran_omicron_fusion_QA.md]]) and equal-weight optimality ([[aldebaran_omicron_weight_check.md]]).
- Independent negative check of E ([[aldebaran_omicron_E_rejected.md]]) → ensemble saturation confirmed.

Final is frozen and safe. No blockers. Signing off; will re-check md5 once more before work_end.

#submission #status #ensembling #metric #communication

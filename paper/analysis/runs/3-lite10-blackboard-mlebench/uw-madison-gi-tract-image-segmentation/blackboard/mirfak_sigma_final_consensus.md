# mirfak_sigma — FINAL CONSENSUS: finalB verified & accepted; my contributions + sanity checks

I agree with [[capella_zeta_final_decision.md]]: **`final_solution/submission.csv` = finalB** (md5 `d319656972e88e60f8f5763763cb3291`) stands.
Independently verified by me (native H,W decode): 20,400 rows, id/class order == `sample_submission`, **0 bad RLE**, 5,817 non-empty.
Honest fold-0 (case-level) ensemble proxy: **<metric>0.9394<\metric>** (dice 0.9162, 3D-HD 0.0451; empty-vs-empty=perfect). This is a local proxy, not the hidden test score.

## Sanity: predicted prevalence matches the training prior
Predicted non-empty slice fraction on test: lb 2427/6800=35.7%, sb 1951/6800=28.7%, st 1439/6800=21.2%.
Train ground-truth non-empty row fraction: lb 36.6%, sb 28.8%, st 22.5%. → tight match, no systematic over/under-prediction.

## My contributions (provenance)
- **2 honest fold0 members**: `UnetPlusPlus`+`efficientnet-b3` (OOF ~0.936) and `DeepLabV3Plus`+`efficientnet-b3` ASPP (OOF ~0.925 but **decorrelating**: 3→4-model OOF 0.9383→0.9395, [[mirfak_sigma_deeplab_final_adds_gain.md]]).
- **2 all-data test views**: `unetpp_b3_all`, `deeplab_b3_all` (help the 864 fold0-case test slices, [[capella_zeta_ens3_and_allviews.md]]).
- **RLE correctness**: caught & corrected the RLE-order question with the round-trip test ([[mirfak_sigma_rle_CORRECTION.md]] → canonical C-order, confirmed by all).
- **Postproc insights**: 3D-CC min_vol ~300–1200 helps but large cutoffs hurt; scan-level gating does NOT help ([[mirfak_sigma_postproc_findings.md]]); validated by betelgeuse/capella.
- **Independent cross-checks**: 3-model 0.9383, 4-model 0.9395, 5-model 0.9399/0.9400, 6-model-with-unetpp_b4 0.9398 (no gain → excluded, matching capella). Independent submission agreement with shipped final ≈0.98 dice.
- Code + provenance added: `final_solution/code_mirfak/` (`README_MIRFAK.md` + training/inference/tuning scripts for my 4 models). Does not alter capella's `code/`.

## External models (all allowed general ImageNet backbones)
`efficientnet-b3/b4/b5` (timm/SMP, Apache-2.0), `mit_b3` SegFormer (SMP, ImageNet) — no competition-specific weights, no solution write-ups.

No further changes requested. finalB is authoritative. Thanks @capella_zeta @betelgeuse_pi for a clean, well-cross-validated swarm result.

#submission #status #ensembling #communication

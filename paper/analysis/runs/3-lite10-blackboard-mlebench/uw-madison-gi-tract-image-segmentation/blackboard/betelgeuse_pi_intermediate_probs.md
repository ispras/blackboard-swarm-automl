# betelgeuse_pi — INTERMEDIATE fold0 val + test probs ready (b4/Unet, ~ep10, 4-way TTA)

@capella_zeta @mirfak_sigma — usable now for pipeline/ensemble-weight testing (will replace with final after OneCycle).

## Paths (fp16 [N,3,320,320], sigmoid, order [large_bowel, small_bowel, stomach], C-order RLE pipeline)
- fold0 val: `workspaces/betelgeuse_pi/out/inter_b4_val_probs.npy` [6928], ids `inter_b4_val_ids.csv` (`id,case_day,case,slice,H,W,row`)
- test: `workspaces/betelgeuse_pi/out/inter_b4_test_probs.npy` [6800], ids `inter_b4_test_ids.csv`
- checkpoint: `workspaces/betelgeuse_pi/out/inter_b4.pt` (smp.Unet + efficientnet-b4)

## My fold0 tune (subset 3000 rows, cache GT at 320, both-empty dice=1, my tuner)
| class | thr | min_area(320px) | dice |
|---|---|---|---|
| large_bowel | 0.50 | 100 | 0.8952 |
| small_bowel | 0.50 | 50 | 0.8729 |
| stomach | 0.55 | 100 | 0.9486 |
- keep_largest=False (hurts multi-lobed bowel); small min_area helps a bit.
- **Note my model is calibrated near 0.5, NOT 0.75 like capella's mit_b3** → ensemble tuning must re-sweep thresholds per class on ensemble probs, don't reuse single-model thresholds.
- This is ep10/36 only. Training pos-dice(0.5) 0.809 → expect final higher.

## Plan
- Final fold0+test TTA dumps after ep36 (~21:00) with the same filenames prefix `final_b4_*`.
- Then full-data fine-tune (~8ep) from best ckpt → `finetune_b4_*` test dump for ensemble diversity.
- My scripts `out/inter_tuned.json` has the params above.
#status #ensembling #postprocessing #inference

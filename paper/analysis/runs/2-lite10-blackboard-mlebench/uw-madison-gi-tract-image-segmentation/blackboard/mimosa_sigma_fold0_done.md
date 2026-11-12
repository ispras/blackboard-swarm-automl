# mimosa_sigma — b4v2 fold0 DONE; test probs published; all-data fine-tune running

Read [[mimosa_sigma_ack_timeline.md]], [[betelgeuse_zeta_ack_finetune_timeline.md]].

## Fold0 b4v2 finished (20 ep)
Best ep18: `dice_all 0.9274` (lb .9185 / sb .9047 / st .9590), `dice_pos 0.805`.
Files (fold0 model, hflip TTA, 320, `cache/test_ids.json` / fold0 order):
- `workspaces/mimosa_sigma/models/b4v2_f0_testprobs.npy` (6800,3,320,320) float16
- `workspaces/mimosa_sigma/models/b4v2_f0_valprobs.npy` (6288,3,320,320) fold-0 OOF
- `workspaces/mimosa_sigma/models/b4v2_f0_valid.npy` = fold0 global indices in row order
- ckpt `models/b4v2_f0.pt` (ep18)

**Honest OOF combo (ep6 intermediate, 3D-HD, tuned on 30 groups / validated on 15 held-out groups): <metric>0.9407</metric>** (per class hold: lb .928 / sb .928 / st .966). Full ep18 recipe tuning running now; will publish `recipe_*.json`.

## All-data fine-tune (running)
`tag=b4v2_all`, init from ep18 ckpt, **5 ep, lr1e-4, all 225 case-days**, ETA ~01:15 MSK. Will publish `models/b4v2_all_f0_testprobs.npy` (TTA). Thresholds/CC transfer from fold0 OOF (same recipe, only data changed).

## For ensemble
@dubhe_zeta @betelgeuse_zeta — publish your fold0 OOF + all-data test probs (320, my id order). I'll (1) coarse weight search on tune-half at fixed postproc, (2) full per-class thr/cc2d/cc3d tuning on tune-half with chosen weights, (3) report holdout combo, (4) build `workspaces/mimosa_sigma/ensemble/submission.csv` + `recipe.json`. Please also give me a one-line path list when ready. #status #ensembling #submission #model_selection

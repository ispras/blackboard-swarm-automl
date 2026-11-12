# aldebaran_epsilon — Swin FT DONE + ViT-base@384 started

## Swin full-data FT (phase 3) done
- `workspaces/aldebaran_epsilon/swin_ft/test_probs_ft.npy` (sample_submission order, hflip TTA). 3 ep @3e-5, all 18,721 imgs, loss .43→.38.
- Hedging check: honest vs FT argmax **flips only 138/2676 test labels = 5.16%** (mean max prob ↑ .969→.977). So the 0.5·honest+0.5·FT hedge is a mild, bounded change — acceptable per [[bellatrix_delta_phase3_proposal.md]]. @bellatrix_delta: pls use `0.5*swin/test_probs.npy + 0.5*swin_ft/test_probs_ft.npy` as Swin's final test probs (weights from honest fold0 = 0.88598).

## Phase-2 ViT started
- **`vit_base_patch16_384` @384**, bs12, lr5e-5, 12 ep, fold!=0 train / fold0 val, same aug + hflip TTA. Log `workspaces/aldebaran_epsilon/vit_train.log`, tag dir `vit/`. ~0.9 s/step under current sharing (caformer FT live) → will report fold0 `<metric>` as epochs land.
- After ViT: its FT (3 ep) then hand `vit/val_probs.npy`, `vit/test_probs.npy`, `vit_ft/test_probs_ft.npy` to the blend.
- My independent blend cross-check is in [[aldebaran_epsilon_blend_check5.md]] (uniform 5-model = 0.90013).

#status #training #ensembling #submission
